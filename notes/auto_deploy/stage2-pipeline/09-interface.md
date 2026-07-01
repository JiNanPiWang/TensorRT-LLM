# 09 — interface.py

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/interface.py`
> 行数: 850
> 问题: Transform 基类和注册机制是什么样的？

---

## 阅读笔记

_TODO_

## 关键结构

```
BaseTransform.__call__ → 检查 enabled → 前置处理 → _apply → 后置验证
TransformRegistry.register("name") → 装饰器注册
TransformConfig → enabled, stage, run_per_gm, requires_shape_prop, ...
TransformInfo → skipped, num_matches, is_clean, has_valid_shapes
```

## 关键点

- [ ] `Stages` enum 有哪些值？
- [ ] `requires_clean_graph` 和 `requires_shape_prop` 什么时候需要？
- [ ] `run_per_gm: false` 是什么意思？

## 疑问

_TODO_
