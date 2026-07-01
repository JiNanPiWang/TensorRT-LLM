# 01 — llm.py：LLM 入口

> 文件: `tensorrt_llm/llmapi/llm.py`（~1900 行）
> 问题: 用户怎么用 TRT-LLM？一条 `LLM("meta-llama/Llama-3.1-8B").generate("Hello")` 发生了什么？

---

## 类继承结构

```
BaseLLM                    ← 后端选择 + 参数解析 + MPI/scheduler 搭建
  └── _TorchLLM            ← PyTorch 后端专用的 _build_model（创建 executor）
        └── LLM            ← 用户直接使用的公开类（只是 _TorchLLM 的别名）
```

`LLM = _TorchLLM`（line 1883），**TensorRT 后端用 `from tensorrt_llm._tensorrt_engine import LLM`**，两者是不同的类。

---

## BaseLLM.__init__（line 240-400）—— 后端选择

核心逻辑在 line 265-282：

```python
backend = kwargs.get('backend', None)
if backend == "pytorch":
    llm_args_cls = TorchLlmArgs
elif backend == '_autodeploy':
    from .._torch.auto_deploy.llm_args import LlmArgs as AutoDeployLlmArgs
    llm_args_cls = AutoDeployLlmArgs
else:
    llm_args_cls = TrtLlmArgs
```

**关键点**：
- **默认走 PyTorch**（backend 没指定时走 `TrtLlmArgs`（TensorRT 遗留后端），但 `_TorchLLM` 会填 `backend="pytorch"`）
- **`backend="_autodeploy"` 激活你的团队代码**：此时候选的 args 类变成 `AutoDeployLlmArgs`，这是一个包含 `transforms`、`compile_backend`、`world_size` 等 AutoDeploy 专属字段的 Pydantic model
- 所有后端使用同一个 `GenerationExecutor` 类（`_executor_cls`），但内部会据 `backend` 字段分叉

### 参数校验

line 284-292：
```python
valid_keys = set(list(llm_args_cls.model_fields.keys()) + ['_mpi_session', 'backend'])
for key in kwargs:
    if key not in valid_keys:
        raise ValueError(...)
```
Pydantic 的 `model_fields` 给出了所有合法参数名，不在其中的直接抛错 —— 保证用户不会乱传参。

### MPI 多 GPU

line 325-344：如果 `parallel_config.is_multi_gpu`，创建 `MpiPoolSession` 或 `MpiCommSession`。多 GPU 用 `mpi4py` spawn 多个 worker 进程。

### 构建模型

line 364：`self._build_model()` —— 虚方法，由子类实现。

---

## _TorchLLM._build_model（line 1784-1860）—— 创建 Executor

这是 PyTorch/AutoDeploy 路径的核心初始化：

```python
def _build_model(self):
    super()._build_model()  # → BaseLLM._build_model：下载/缓存模型到 _hf_model_dir
    assert self._engine_dir is None  # PyTorch 后端没有 engine 概念

    # 1. 加载 tokenizer + HF config
    self._tokenizer = self._try_load_tokenizer()
    self._hf_model_config = self._try_load_hf_model_config()

    # 2. 处理多模态（如果有）
    self.input_processor = create_input_processor(...)

    # 3. 创建 executor —— 这是关键
    self._executor = self._executor_cls.create(
        llm_args=self.args,
        hf_model_dir=self._hf_model_dir,
        tokenizer=self.tokenizer,
        ...
    )
```

`GenerationExecutor.create()` 内部：

```text
_create_ipc_executor()
  → worker_main()         ← 新进程入口
    → GenerationExecutorWorker.__init__()
      → BaseWorker.setup_engine()    ← 看这里！
```

**`BaseWorker.setup_engine()`（`base_worker.py:151-234`）是决定创建什么引擎的分叉点**：

```python
if self._backend == "pytorch":
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import create_py_executor
    create_executor = create_py_executor
elif self._backend == "_autodeploy":
    from tensorrt_llm._torch.auto_deploy.shim.ad_executor import create_autodeploy_executor
    create_executor = create_autodeploy_executor
```

**这就是 PyExecutor 和 ADExecutor 的分叉**。AutoDeploy 用 `ADExecutor`（继承 `PyExecutor`），在 `PyExecutor` 的基础上叠加了图优化 pipeline。**调度、KV Cache、CUDA Graph 等所有运行时基础设施都是相同的**，只有"模型怎么执行"不同。

---

## generate 系列（line 491-703）

### generate（line 491-588）—— 同步模式

```python
def generate(self, inputs, sampling_params=None, ...):
    # 1. 把单个输入包装成列表
    if unbatched:
        inputs = [inputs]

    # 2. 对每个输入调用 generate_async
    futures = []
    for i, request_input in enumerate(...):
        future = self.generate_async(request_input, ...)
        futures.append(future)

    # 3. 阻塞等待所有 future
    for future in tqdm(futures):
        future.result()  # 这里阻塞

    return futures  # 或单个（如果 unbatched）
```

本质是 `generate_async` 的同步封装 + tqdm 进度条。

### generate_async（line 591-703）—— 异步模式

核心链路：

```python
def generate_async(self, inputs, sampling_params=None, ...):
    # 1. CPU 侧预处理（tokenization, 多模态）
    (prompt_token_ids, prompt, query_token_ids,
     multimodal_params, encoder_input_token_ids) = self._preprocess(...)

    # 2. 调用 executor 的 generate_async
    result = self._executor.generate_async(
        prompt_token_ids,
        query_token_ids=query_token_ids,
        sampling_params=sampling_params,
        ...
    )

    # 3. 包装成 RequestOutput
    return RequestOutput._from_generation_result(result, prompt, self.tokenizer, ...)
```

**关键**：`self._executor.generate_async()` 发请求给 worker 进程（通过 IPC/ZMQ），worker 内部的 `PyExecutor` 接收并调度执行。请求和响应的通信是通过 `FusedIpcQueue`（共享内存 + 信号）实现的。

---

## 总结：完整请求路径

```
用户: LLM("meta-llama/Llama-3.1-8B").generate("Hello")
  │
  ├─ __init__ 阶段（只执行一次）
  │   ├─ BaseLLM.__init__: 解析参数 → TorchLlmArgs（或 AutoDeployLlmArgs）
  │   ├─ _TorchLLM._build_model:
  │   │   ├─ 下载/加载 HF 模型权重
  │   │   ├─ 创建 tokenizer
  │   │   └─ GenerationExecutor.create()
  │   │       └─ spawn worker 进程
  │   │           └─ BaseWorker.setup_engine()
  │   │               ├─ pytorch → create_py_executor() → PyExecutor
  │   │               └─ _autodeploy → create_autodeploy_executor() → ADExecutor
  │   │
  │   └─ executor 就绪，等待请求
  │
  └─ generate("Hello") 阶段
      ├─ tokenize "Hello" → [token_ids]
      ├─ _executor.generate_async(token_ids, sampling_params)
      │   └─ IPC 发送到 worker 进程
      │       └─ PyExecutor.enqueue_request()
      │           ├─ Scheduler 打包成 batch
      │           ├─ ModelEngine.forward() → logits
      │           ├─ Sampler → 选中 token
      │           └─ IPC 返回 result
      └─ RequestOutput(prompt="Hello", outputs=[CompletionOutput(" world")])
```

### 你的团队（AutoDeploy）跟标准 PyTorch 后端的唯一区别

在 `BaseWorker.setup_engine()` 分叉以后：
- PyTorch：`create_py_executor` → 直接创建 `PyExecutor`
- AutoDeploy：`create_autodeploy_executor` → 创建 `ADExecutor(PyExecutor)`，**在模型 forward 之前插入了 FX Graph → Transform → Compile 的 pipeline**

一切其他代码（LLM API、executor IPC、scheduler、KV cache）都是二组共用的。

### 按你之前的面试内容关联

| 面试题目 | 对应这里的 |
|---------|-----------|
| torch.compile 怎么做 | `torch.compile` 在 Stage 5 的编译阶段，替换模型 forward |
| CUDA Graph 怎么 capture | 在 `PyExecutor`（Stage 1）的 `CudaGraphRunner` 里 |
| attention 怎么优化 | Stage 6 的 attention fusion（`fuse_rope_into_trtllm_attention`） |
| AI 编译器怎么做图变换 | Stage 4 的 `BaseTransform` + `inserting_before()` placement |
