# 10 — ad_executor.py：AutoDeploy 的执行引擎

> 文件: `tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py`（~1335 行）
> 问题: AutoDeploy 怎么和 PyExecutor 对接？InferenceOptimizer 在什么时候被调用？

---

## 核心纠正：没有 ADExecutor 类

之前的笔记说"ADExecutor 继承 PyExecutor"——**这是错的**。实际架构是：

```python
# ❌ 不存在这个：
class ADExecutor(PyExecutor): ...

# ✅ 实际的：
def create_autodeploy_executor(ad_config, ...):
    # 1. 创建 ADEngine（包含编译后的模型）
    engine = ADEngine.build_from_config(ad_config, ...)

    # 2. 创建标准的 PyExecutor，但 model_engine 用的是 ADEngine
    py_executor = PyExecutor(
        resource_manager, scheduler,
        model_engine=engine,   # ← ADEngine，不是 PyTorchModelEngine
        sampler, dist, ...,
    )
    return py_executor
```

**也就是说，整个 TRT-LLM 只有一个 PyExecutor 类**。PyTorch 后端和 AutoDeploy 后端的区别仅在 `model_engine` 参数：
- PyTorch 后端：传 `PyTorchModelEngine`（加载原始 HF 模型，用 torch.compile 包裹）
- AutoDeploy 后端：传 `ADEngine`（全量 FX Graph + Transform + torch.compile 的编译产物）

---

## ADEngine.build_from_config（line 374-425）—— 核心入口

```python
@classmethod
def build_from_config(cls, ad_config, dist_config, mapping, dist):
    # 1. 创建 ModelFactory（管理 HF 模型加载 + 推理配置）
    factory = ad_config.create_factory()

    # 2. 创建 CachedSequenceInterface（KV cache 管理 + sequence 状态）
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=ad_config.max_seq_len,
        max_batch_size=ad_config.max_batch_size,
        kv_cache_config=ad_config.kv_cache_config,
        vocab_size_padded=factory.vocab_size_padded,
        ...
    )

    # 3. 🎯 创建 InferenceOptimizer — 这里触发全量 transform pipeline！
    build_and_optimize = InferenceOptimizer(
        factory=factory,
        config=ad_config.transforms,  # ← 来自 default.yaml + 用户覆盖
        dist_config=dist_config,
    )

    # 4. 创建 ADEngine
    return cls(build_and_optimize, cache_seq_interface, ...)
```

### ADEngine.__init__（line 427-460）

```python
def __init__(self, get_inference_model, cache_seq_interface, ...):
    # get_inference_model 实际是 InferenceOptimizer.__call__
    # 在 ADEngine._forward_init() 中懒加载调用，而不是在 __init__ 中
    self._get_inference_model = get_inference_model
    self.cache_seq_interface = cache_seq_interface
```

### 懒加载：模型在第一次 forward 时才编译

`ADEngine` 不在 `__init__` 中跑 transform pipeline，而是在 `_run_forward()` 第一次被调用时才触发 `InferenceOptimizer(cache_seq_interface)` —— **因为 KV cache 的配置需要在 cache_init 阶段才能最终确定**。

---

## ADEngine.forward（line 1038-1080）

```python
def forward(self, scheduled_requests, resource_manager, ...):
    # 1. 把 TRT-LLM 的 ScheduledRequests 转换成 AD 内部的 SequenceInfo
    self._prepare_inputs(scheduled_requests, resource_manager, ...)

    # 2. 运行编译后的模型
    outputs = self._run_forward()

    # 3. 后处理（logit 处理）
    if self.dist_config is not None:
        self._execute_logit_post_processors(scheduled_requests, outputs)

    return outputs
```

### 和 PyTorchModelEngine.forward 的关键区别

| | PyTorchModelEngine | ADEngine |
|---|---|---|
| 模型 | 原始 HF 模型 + torch.compile | FX Graph → Transform → torch.compile |
| KV cache 管理 | 手动：block_tables 传给 attention metadata | 由 CachedSequenceInterface 统一管理 |
| 输入转换 | 手写 `_prepare_model_inputs` | 由 export 时的 placeholder 自动匹配 |
| CUDA Graph | CUDAGraphRunner | piecewise CUDA Graph（compile 阶段注入） |
| 调度接口 | 直接读 `scheduled_requests.xxx_requests` | 通过 `_prepare_inputs` 到 `SequenceInfo` |

---

## 数据流总结

```
create_autodeploy_executor(ad_config)
  ├─ ADEngine.build_from_config(ad_config, dist_config)
  │   ├─ factory = ad_config.create_factory()          ← ModelFactory 管理 HF 加载
  │   ├─ cache_seq_interface = CachedSequenceInterface(...)  ← KV cache 接口
  │   ├─ build_and_optimize = InferenceOptimizer(
  │   │       factory, config=ad_config.transforms)
  │   └─ ADEngine(build_and_optimize, cache_seq_interface, ...)
  │
  ├─ kv_cache_manager = engine.cache_seq_interface.kv_cache_manager  ← transform 已创建
  ├─ resource_manager = ResourceManager(kv_cache_manager, seq_slot_manager)
  ├─ scheduler = SimpleScheduler(capacitor_scheduler, mb_scheduler)
  ├─ sampler = instantiate_sampler(ad_config, ...)
  │
  └─ PyExecutor(resource_manager, scheduler, model_engine=engine, sampler, dist)
       └─ 标准 PyExecutor 事件循环，
          但 engine.forward() 调用的是编译后的 FX Graph
```

---

## 总结

- **没有 ADExecutor 类**。AutoDeploy 直接用 `PyExecutor`，区别在 `model_engine`（ADEngine vs PyTorchModelEngine）
- **ADEngine = PyExecutor 的 ModelEngine 接口** + **InferenceOptimizer（transform pipeline）** + **CachedSequenceInterface（KV cache 管理）**
- Transform pipeline 在第一次 forward 时懒加载触发，不是 init 时
- `InferenceOptimizer(factory, config=ad_config.transforms)` 中的 `ad_config.transforms` 来自 `default.yaml` + 用户自定义覆盖


---
**下一节：[`11 — 图导出`](../stage3-export/11-export.md)**
