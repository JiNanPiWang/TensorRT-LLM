# 14 — fuse_silu_mul.py（最佳 Fusion 模板）

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/library/fuse_silu_mul.py`
> 行数: 406
> 问题: 一个完整的 fusion transform 长什么样？从头到尾的模式是什么？

---

## 阅读笔记

_TODO_

## 融合前后

```
Before: silu(narrow(x,0,N)) * narrow(x,N,N)  →  2 narrow + silu + mul
After:  silu_and_mul(x)                        →  1 fused op
```

## 核心函数

| 函数 | 作用 |
|------|------|
| `_try_fuse_mul()` | 从 mul 开始匹配 pattern |
| `_match_silu_narrow_mul()` | 验证 silu(narrow(x,0,N)) * narrow(x,N,N) |
| `_get_narrow_info()` | 提取 (parent, offset, length) |
| `_strip_contiguous()` | 穿透 .contiguous() 包装 |
| `_try_fuse_fp8_quant()` | FP8 量化折叠进 kernel |

## 为什么这是最佳模板

```
1. Pydantic Config → FuseSiluMulConfig
2. @TransformRegistry.register("fuse_silu_mul")
3. _apply: 检查 enabled → 遍历节点 → 匹配 pattern
4. inserting_before → call_function → replace_all_uses_with
5. eliminate_dead_code + recompile
6. 额外优化 pass（FP8 折叠）
```

## 关键点

- [ ] 为什么只在 parent.shape[-1] == 2 * half_size 时才融合？
- [ ] `flashinfer` vs `trtllm` backend 的区别？
- [ ] FP8 量化折叠怎么改写下游 linear 的参数？

## 疑问

_TODO_
