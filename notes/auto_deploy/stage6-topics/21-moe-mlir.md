# 21 — MoE + MLIR 融合

> 文件: `auto_deploy/transform/library/fused_moe.py`,
>       `auto_deploy/transform/library/mlir_elementwise_fusion.py`
> 方向: MoE 权重合并 + MLIR 自动融合

---

## 阅读笔记

_TODO_

## MoE 关键点

- [ ] MoE export 优化（trace 时减少 experts）
- [ ] FusedMoE 的权重布局
- [ ] all-to-all 通信的角色

## MLIR 关键点

- [ ] Pipeline: FX → MLIR (xDSL) → Decompose → Discover → Triton → FX
- [ ] 什么时候用 MLIR 融合 vs 手写 fusion？

## 疑问

_TODO_
