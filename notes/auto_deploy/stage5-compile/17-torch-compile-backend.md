# 17 — torch-compile 编译后端

> 文件: `compile/compiler.py`（57行），`compile/backends/`（4 个后端），
> `compile/backends/torch_cudagraph.py`（1279行 — 核心）
> 问题: AutoDeploy 怎么把融合后的图交给 torch.compile + CUDA Graph？

---

## 后端注册表

```python
class CompileBackendRegistry:
    _backend_registry = {}

@CompileBackendRegistry.register("torch-compile")
class TorchCompileCompiler:    # torch.compile only, 无 CUDA Graph

@CompileBackendRegistry.register("torch-cudagraph")
class TorchCudagraphCompiler:  # 默认！torch.compile + CUDA Graph

@CompileBackendRegistry.register("torch-opt")
class TorchOptCompiler:        # torch.compile + CUDA Graph（优化版）

@CompileBackendRegistry.register("torch-simple")
class TorchSimpleCompiler:     # 无 torch.compile, 纯 CUDA Graph
```

---

## 默认后端：torch-cudagraph（Dual Mode）

```python
class TorchCudagraphCompiler:
    def compile(self):
        # 1. Monolithic CUDA Graph（decode-only batch）
        monolithic = CapturedGraph(target_gm, ...)
        monolithic.capture_graph(get_args_for_batch_size, batch_sizes)

        # 2. Piecewise CUDA Graph（prefill/mixed batch）
        if self.piecewise_enabled:
            piecewise = PiecewiseCapturedGraph(model=target_gm, ...)
            piecewise.prepare()           # 在动态 op 处切分模型
            piecewise.warmup_and_capture(get_piecewise_args)

        # 3. 快速 path 分发
        return DualModeCapturedGraph(monolithic, piecewise)
```

### DualModeCapturedGraph.forward（line 1037-1071）

```python
def forward(self, *args, **kwargs):
    if self._is_decode_only(**kwargs):
        return self.monolithic(*args, **kwargs)     # ← full graph replay
    # prefill/mixed batch
    num_tokens = self._get_num_tokens(**kwargs)
    bucket = self._find_nearest_bucket(num_tokens)   # ← 找最近的更大 bucket
    if bucket is not None:
        result = self.piecewise(*args, num_tokens=bucket, **kwargs)
        if bucket > num_tokens:
            result = self._truncate_output(result, num_tokens, bucket)  # ← 切到真实尺寸
        return result
    return self.piecewise.original_model(*args, **kwargs)  # eager fallback
```

**核心设计**：
- **Decode-only** → Monolithic CUDA Graph（整个 forward 一次 replay）
- **Prefill/Mixed** → Piecewise（在动态 op 边界切图，静态段各自 capture，动态段跑 eager）
- **Bucket padding** → 如果 `num_tokens=150`，最近的 bucket 是 `256`，padding 到 256 后 replay，输出截断回 150

---

## Piecewise CUDA Graph（line 465-907）

在动态 op（attention, SSM, cache insert 等）处切分 FX Graph：

```python
split_gm = split_graph_at_dynamic_ops(gm)
# 结果：submod_0 (static) → submod_1 (dynamic) → submod_2 (static) → ...
```

- **Static submodules** → 被 `ADPiecewiseRunner` 包装，各自 capture 为独立的 CUDA Graph
- **Dynamic submodules** → 被 `DynamicOpWrapper` 包装（需要 output buffer 的）或跑 eager

---

## `dynamic=True` 在推理场景的含义

```python
torch.compile(self.model, dynamic=True)
```

- `dynamic=True`：允许输入 shape 变化时重用编译缓存
- `recompile_limit`：torch.compile 最多重编译多少次（需 ≥ batch_sizes 数量）
- **推理场景的关键**：不同 batch_size 可能触发不同的 recompilation，需要 `dynamic=True` + 足够的 `recompile_limit`

---

## 叠加顺序

```
1. torch.compile(model, dynamic=True)  ← 先把 Python 代码编译成 Triton/C++ kernel
     ↓
2. CUDA Graph capture                ← 然后把 kernel launch 序列录制成 graph
     ↓
3. CUDA Graph replay                 ← 推理时一次 replay 替代 N 次 kernel launch
```

**torch.compile 解决"kernel 不够快"的问题，CUDA Graph 解决"kernel launch 太多"的问题。两者互补。**

---

## 你的融合和编译后端的关系

你的 fusion（Stage 4）发生在 **torch.compile 和 CUDA Graph 之前**：
- Fusion 简化图结构 → torch.compile 更容易生成优质 kernel
- Fusion 融合小 op → CUDA Graph 中更少的分段 → 更少的 graph pieces → 更低的开销
