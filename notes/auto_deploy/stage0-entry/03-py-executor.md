# 03 — py_executor.py：运行时引擎

> 文件: `tensorrt_llm/_torch/pyexecutor/py_executor.py`（~3500 行）
> 问题: PyExecutor 的事件循环长什么样？请求从进队到出结果经过哪些阶段？

---

## 整体架构

`PyExecutor` 是整个 PyTorch 后端的运行时心脏。它是一个**单线程事件循环**（跑在 worker 进程中），每个 iteration 执行：

```
schedule → forward → sample → update KV cache → respond
```

```
外部（LLM.generate_async）
  │ IPC enqueue
  ▼
executor_request_queue ──► _executor_loop()
                              │
                              ├─ _prepare_and_schedule_batch()  ← Scheduler 打包 batch
                              ├─ _run_encoder_step()            ← encoder-decoder 模型的 encoder 部分
                              ├─ model_engine.forward()         ← GPU 推理
                              ├─ sampler.sample()               ← 从 logits 采样 token
                              ├─ kv_cache_manager.update()      ← 保存新 KV
                              └─ _send_responses()              ← IPC 返回结果
```

---

## PyExecutor.__init__（line 415-505）

构造函数接收的核心依赖：

| 参数 | 类型 | 职责 |
|------|------|------|
| `resource_manager` | `ResourceManager` | GPU 显存管理、CUDA Stream、CUDA Graph 资源 |
| `scheduler` | `RequestScheduler` | Continuous batching：把请求打包成 batch |
| `model_engine` | `ModelEngine` | 执行 `model.forward()` |
| `sampler` | `Sampler` | 从 logits 采样下一个 token |
| `dist` | `Distributed` | 多 GPU 通信抽象 |
| `drafter` | `Drafter` | 投机解码的 draft model |
| `guided_decoder` | `GuidedDecoder` | 结构化输出（grammar 约束） |

**运行时状态变量**：

```python
self.active = True             # 事件循环是否还在跑
self.iter_counter = 0          # 当前 iteration 编号
self.execution_stream          # 独立的 CUDA stream（与通信 stream 分离，避免不必要的同步）
self.encoder_stream            # encoder-decoder 模型的 encoder 专用 stream
self.response_cv               # Condition Variable：保护响应队列的并发
self.active_requests = {}      # 当前活跃的请求
self.waiting_queue = []        # 等待调度的请求
```

**注意** `self.execution_stream` —— 这是 CUDA Graph 能够工作的关键：forward 的所有 kernel launch 都发射到这个 stream 上，CUDA Graph capture/replay 也是在这个 stream 上进行的。另外，encoder stream 独立出来是为了 encoder-decoder 模型：encoder forward 不应该阻塞 decoder forward。

---

## enqueue_request（line 1226-1238）

```python
def enqueue_request(self, request, query=None, result_wait_queue=None):
    req_id = self.executor_request_queue.enqueue_request(request, query)
    if result_wait_queue is not None:
        with self.response_cv:
            self.result_wait_queues[req_id] = result_wait_queue
    return req_id
```

非常薄的一层：把请求塞进 `executor_request_queue`。真正的调度和執行在 `_executor_loop` 里。

---

## _executor_loop（line 3117-3540）—— 核心事件循环

```python
def _executor_loop(self):
    torch.cuda.set_device(self.device_id)
    while True:
        self.hang_detector.checkpoint()          # 防卡死检测

        # ① Schedule：Scheduler 打包 batch + 分配 KV cache 页
        scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
        if scheduled_batch is None:
            break

        # ② Encoder step（如果有 encoder-decoder 请求）
        if scheduled_batch.encoder_requests:
            self._run_encoder_step(scheduled_batch.encoder_requests)

        # ③ 检查是否可以执行 forward（KV cache 资源够吗？）
        can_queue, _ = self._can_queue(scheduled_batch)
        if not can_queue:
            self._revert_gen_alloc(scheduled_batch)  # 回退 KV cache 分配
            continue

        # ④ Forward：ModelEngine 执行模型前向推理
        #    有子循环处理投机解码、guided decoding
        if can_queue:
            self.resource_manager.prepare_resources(scheduled_batch)
            # ... guided_decoder, drafter 设置 ...
            self.model_engine.forward(...)         # ← 这是真正跑 GPU 的地方！

        # ⑤ Sample：从 logits 采样下一个 token
        sample_state = self.sampler.sample(...)

        # ⑥ Update KV cache：把本轮计算的新 KV 写入 cache
        self.kv_cache_manager.update(...)

        # ⑦ Respond：把结果通过 IPC 发回主进程
        self._send_responses(finished_requests)

        self.iter_counter += 1
```

### 三个变体

PyExecutor 有 3 个 `_executor_loop` 变体，选择逻辑在 `_start_executor_loop`：

| 变体 | 条件 | 特点 |
|------|------|------|
| `_executor_loop()` | 默认 | 标准的 step-by-step |
| `_executor_loop_overlap()` | `!disable_overlap_scheduler` | GPU 计算和 CPU 调度重叠 |
| `_executor_loop_pp()` | pipeline_parallel_size > 1 | PP 异步 microbatch |

---

## AutoDeploy 的关系：ADExecutor

`ADExecutor` 在 `_torch/auto_deploy/shim/ad_executor.py` 定义，**继承 `PyExecutor`**。

**因为** `PyExecutor` 提供了完整的调度、KV cache、sampling、IPC 通信基础设施，**所以** `ADExecutor` 只需要覆盖"模型怎么执行"这一部分。具体来说：

- `ADExecutor.__init__` 会创建 `InferenceOptimizer`（图优化 pipeline），用它生成的编译后模型替换 `ModelEngine` 中的原始 HF 模型
- 事件循环、调度、KV cache 管理 —— 全部原封不动继承

---

## 关键接口（外界怎么跟 PyExecutor 交互）

| 方法 | 调用方 | 说明 |
|------|--------|------|
| `enqueue_request()` | `BaseWorker` | 把请求放入队列 |
| `await_responses()` | `BaseWorker`（在单独线程） | 拉取完成的响应，通过 IPC 发回主进程 |
| `shutdown()` | `BaseWorker` | 设置 shutdown flag + 等待事件循环退出 |
| `control_action()` | `BaseWorker` | 动态控制（如 pause/resume scheduler） |

---

## 总结

- PyExecutor 是**单线程事件循环**，每个 iteration = schedule → forward → sample → update cache → respond
- 核心依赖：`Scheduler`（打包 batch）、`ModelEngine`（GPU forward）、`Sampler`（采样）、`KvCacheManagerV2`（KV cache）
- ADExecutor 继承 PyExecutor，只覆盖模型加载部分（用编译图替代 eager forward），其他全部复用
- 事件的 `execution_stream` 和 CUDA Graph 密切相关 —— CUDA Graph capture/replay 都在这个 stream 上进行
