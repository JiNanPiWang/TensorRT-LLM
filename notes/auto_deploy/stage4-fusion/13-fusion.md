# 13 — fusion.py：GEMM 融合核心

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py`（693 行）
> 问题: 怎么把 gate_proj(x) 和 up_proj(x) 两个 GEMM 融合成一个 GEMM？

---

## 为什么需要 GEMM 融合

SwiGLU 结构的 MLP 需要两个 GEMM 共享同一输入：

```python
# 原始
gate = gate_proj(x)    # [batch, dim] → [batch, inter]
up = up_proj(x)        # [batch, dim] → [batch, inter]
hidden = silu(gate) * up

# 融合后 (fuse_gemms)
gate_up = gate_up_proj(x)    # [batch, dim] → [batch, 2*inter]
# 权重: cat([gate_proj.weight, up_proj.weight], dim=0)
# 然后由 fuse_silu_mul 处理 narrow + silu + mul
```

---

## 融合条件（`check_same_children`）

两个 GEMM 能被融合需满足：

1. **parent 相同**：两个 linear 节点的输入来自同一个节点
2. **dtype 一致**：`input.dtype == weight.dtype`
3. **bias 状态一致**：要么都有 bias 要么都没有
4. **weight shape 兼容**：除 output dim 外，其他维度相同
5. **量化 scale 匹配**（FP8/FP4）：`input_scale` 和 `weight_scale` 相同
6. **所有的 consumer 都是同一类型**（对 FuseGemms）：没有非 linear 的 consumer

---

## `_insert_fused_gemm()` — placement 策略

这是你面试中讨论的关键工程决策：

```python
with gm.graph.inserting_before(linear_nodes[0]):
    fused = gm.graph.call_function(
        torch.ops.auto_deploy.fused_linear.default,
        args=(parent_node, fused_weight, fused_bias),
    )
```

**为什么放在 `linear_nodes[0]` 之前而不是之后？**

- 放在第一个被融合 GEMM 的**前面** → 保证融合后的 GEMM 在所有原始 consumer 之前执行
- 如果放在**最后面** → 可能会有其他 consumer 读到的数据在融合 GEMM 修改后被污染
- 这是一个 FX Graph 拓扑排序的约束

---

## FuseGemms vs FuseGemmsMixedChildren

| | FuseGemms | FuseGemmsMixedChildren |
|---|---|---|
| 条件 | parent 的所有 consumer **必须全部是** linear | parent 的 consumer **可以混合**（linear + non-linear） |
| 融合方式 | 全部 consumer 的权重 concat | 只融合 linear consumer，non-linear 保持原样 |
| 适用场景 | `gate_proj + up_proj`（两个都是 linear） | parent 同时被 linear 和 add/skip 使用时 |

---

## 为什么默认 disabled

```yaml
fuse_gemms:
  enabled: false  # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 OOM
```

FuseGemms 会增加中间结果的内存占用（`2*intermediate_size` vs 两个 `intermediate_size`），在某些情况下导致 OOM。**你的工作之一可能就是分析 OOM 原因，找到内存安全的融合策略然后默认开启**。

---

## FP8/FP4/FineGrained 的特殊处理

量化 GEMM 融合比 FP16 复杂——因为融合要共享 scale：

```python
# FP8: input_scale 和 weight_scale 必须匹配
# FP4: block_size 和 alpha 必须匹配
# FineGrained: 逐行 scale 维度必须匹配
if scale_mismatch:
    return  # 不融合
```

量化参数不匹配时不能融合，否则精度会下降。这是一种 **安全第一** 的设计。

---

## 动态 scale（`allow_different_input_scales`）

对于 `fuse_fp8_moe`，有一个特殊选项：

```yaml
fuse_fp8_moe:
  allow_different_input_scales: false  # 默认 false：不同 scale 不融合
```

如果设为 true，融合时可以使用不同的 input_scale（但需要额外的 scale 处理逻辑）。


---
**下一节：[`14 — SiLU+Mul 融合模板`](../stage4-fusion/14-fuse-silu-mul.md)**
