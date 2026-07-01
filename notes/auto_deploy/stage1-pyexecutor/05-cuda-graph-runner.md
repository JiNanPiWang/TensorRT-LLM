# 05 — cuda_graph_runner.py：CUDA Graph

> 文件: `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`（~1100 行）
> 问题: piecewise CUDA graph 怎么 capture 和 replay？为什么要 piecewise？

---

## 为什么需要 CUDA Graph

LLM 推理的 decode 阶段：
- 每次只处理 **1 个 token**（per request）
- GPU kernel launch 开销占总时间比例很高（几百个 kernel，每个几微秒）
- CUDA Graph 把 **一整个 forward pass 的 kernel 序列录制成一个 graph**，replay 时只需 1 次 kernel launch

**但 prefill 阶段不能用**：prefill 的输入长度变化大，没法预先 capture 所有可能长度。

---

## Full vs Piecewise CUDA Graph

| | Full CUDA Graph | Piecewise CUDA Graph |
|---|---|---|
| Capture 粒度 | 整个 forward pass（input → output） | 每个 transformer layer 一个 graph |
| 优点 | 零 kernel launch 开销 | 可以处理不同 batch_size / input 变化 |
| 缺点 | 任何输入尺寸变化都导致 fallback | 每个 layer 之间仍有少量 CPU 开销 |
| TRT-LLM 默认 | 否 | **是**（使用 `piecewise_cuda_graph` 装饰器） |

### Piecewise 怎么工作

```python
@piecewise_cuda_graph
def forward(self, x, attn_metadata, ...):
    for layer in self.layers:
        x = layer(x, attn_metadata)   # ← 每个 layer 独立 capture 为一个小 CUDA Graph
    return x
```

每个 decoder layer 被捕获为独立的 CUDA Graph。**好处**是：batch_size 变化只影响 attention 和 MLP 的 graph shape，不同层之间可以复用。

---

## CUDAGraphRunnerConfig：三种执行路径

```python
@dataclass
class CUDAGraphRunnerConfig:
    use_cuda_graph: bool  # 总开关
```

`use_cuda_graph` 决定三种路径：

1. **`False`（纯 eager）**：所有 forward 走 eager
2. **`True`（混合模式）**：decode batch → CUDA Graph；prefill batch → eager fallback
3. **`True`（纯 CUDA Graph）**：走 graph，没命中则 fallback eager

---

## Capture 流程（在 warmup 阶段）

`model_engine.py` 的 `warmup()` 调用：

```python
# 对每个 batch_size 和 num_tokens 组合
for bs in batch_sizes:
    for num_tokens in capture_num_tokens:
        dummy_inputs = _build_dummy_inputs(bs, num_tokens)
        cuda_graph_runner.capture(
            key=(bs, num_tokens, ...),  # KeyType
            forward_fn,
            dummy_inputs,
        )
```

Key 的结构：`(batch_size, num_tokens, is_context, enable_attention_dp, enable_capture_debug)`

Replay 时根据实际 batch 找最近的大 key（padding 到 capture 的 batch_size）。

### Batch padding

如果当前 batch_size = 3 但没有 key=3 的 CUDA Graph，但有 key=4：
- 把 batch padding 到 4（加 dummy requests）
- 用 key=4 的 graph replay
- 把 dummy 的输出丢弃

---

## 和 AutoDeploy 的关系

AutoDeploy 的 `compile_backend="torch-cudagraph"` → 在模型编译后仍然使用相同的 CUDA Graph 机制。**编译后端决定的是"模型内部怎么跑"，CUDA Graph 决定的是"kernel launch 怎么录"，两者正交。**

torch.compile 可以在 capture 的 CUDA Graph 内部，也可以在外部——关键是 CUDA Graph capture 发生在 torch.compile warmup 之后。


---
**下一节：[`06 — KV Cache 管理`](../stage1-pyexecutor/06-kv-cache-manager.md)**
