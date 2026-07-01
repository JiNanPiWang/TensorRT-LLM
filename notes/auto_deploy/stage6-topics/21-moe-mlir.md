# 21 — MoE + MLIR 融合

> 文件: `transform/library/fused_moe.py`, `transform/library/mlir_elementwise_fusion.py`
> 方向: MoE 权重合并 + MLIR 自动融合

---

## MoE Fusion

```python
# 原始：每个 expert 独立的 GEMM
for expert_idx in range(num_experts):
    output[expert_idx] = expert_proj[expert_idx](x)

# FusedMoE: 所有 expert 合并为一个融合 kernel
output = fused_moe(x, all_expert_weights, routing_weights)
```

FusedMoE 在 `fuse_moe`、`fuse_fp8_moe`、`fuse_nvfp4_moe` 等 transform 中实现。

## MoE Export 优化

export 阶段只 trace 2 个 expert，然后展开（在 `11-export.md` 中已详细说明）。

---

## MLIR Elementwise Fusion（`mlir_elementwise_fusion`）

```yaml
mlir_elementwise_fusion:
  stage: post_load_fusion
  enabled: false  # 默认关闭
  bypass_ops: []  # 可以指定跳过某些 op
```

### Pipeline

```
FX Graph → MLIR (xDSL) → Decompose → Discover → Triton CodeGen → FX
```

MLIR 在这里的角色是**自动发现可以融合的 element-wise 操作链**，生成融合后的 Triton kernel。区别于手写的融合 transform（如 `fuse_silu_mul`），MLIR fusion 是自动化的。

### 什么时候用 MLIR vs 手写

| | 手写 Fusion Transform | MLIR Elementwise Fusion |
|---|---|---|
| 模式覆盖 | 特化的（SiLU+Mul、SwiGLU） | 通用的（任意 element-wise 链） |
| 开发成本 | 需要为每种 pattern 写代码 | 自动发现 |
| 性能 | 最优（手调） | 接近最优 |
| 当前状态 | 默认开启（稳定） | 默认关闭（实验性） |
