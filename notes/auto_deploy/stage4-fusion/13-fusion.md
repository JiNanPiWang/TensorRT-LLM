# 13 — fusion.py（GEMM 融合核心）

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py`
> 行数: 693
> 问题: GEMM 融合的条件、placement 策略、多输入多输出怎么处理？

---

## 阅读笔记

_TODO_

## 核心函数/类

| 函数/类 | 注册名 | 作用 |
|---------|--------|------|
| `_insert_fused_gemm()` | — | 合并多个共享输入的 GEMM |
| `check_same_children()` | — | 检查 parent 的所有 consumer 是否同类型 |
| `FuseGemms` | `fuse_gemms` | 标准 GEMM 融合（纯 linear children） |
| `FuseGemmsMixedChildren` | `fuse_gemms_mixed_children` | 宽松版（允许非 linear children） |
| `FuseFP8Gemms` | `fuse_fp8_gemms` | FP8 量化 GEMM 融合 |
| `FuseFP4Gemms` | `fuse_fp4_gemms` | NVFP4 量化 GEMM 融合 |
| `FuseFineGrainedFP8Gemms` | `fuse_finegrained_fp8_gemms` | 细粒度 FP8 GEMM 融合 |

## 核心关注点：融合 placement

```python
# fusion.py:147 — 融合后节点放在第一个被融合 GEMM 之前
with gm.graph.inserting_before(linear_nodes[0]):
    fused = gm.graph.call_function(...)
```

## 关键点

- [ ] 融合条件：dtype 一致、bias 状态一致、parent 相同、scale 匹配
- [ ] `allow_not_contiguous` 控制什么？（narrow view vs contiguous copy）
- [ ] `FuseGemms` vs `FuseGemmsMixedChildren` 的区别？
- [ ] 为什么这些 fusion 默认 `enabled: false`？（issue 4674 — OOM）
- [ ] FP8/FP4/FineGrained 各自的 scale/alpha/block_size 不匹配怎么处理？

## 疑问

_TODO_
