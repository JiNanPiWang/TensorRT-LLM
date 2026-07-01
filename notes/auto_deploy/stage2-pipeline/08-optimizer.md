# 08 — optimizer.py

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/optimizer.py`
> 行数: 144
> 问题: InferenceOptimizer 怎么驱动所有 transform？

---

## 阅读笔记

_TODO_

## 关键点

- [ ] `__call__` 的核心循环：遍历 config → 创建 transform → 执行
- [ ] `_clean_config` 怎么按 stage 排序？
- [ ] `_maybe_restore_from_cache` 的 pipeline cache 机制？

## 疑问

_TODO_
