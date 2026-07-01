# 03 — py_executor.py

> 文件: `tensorrt_llm/_torch/pyexecutor/py_executor.py`
> 行数: ~1800
> 问题: PyExecutor 的主循环长什么样？

---

## 阅读笔记

_TODO_

## 关键点

- [ ] PyExecutor 的事件循环：schedule → forward → sample → update cache → respond
- [ ] `__init__` 里怎么创建 Scheduler、ModelEngine、Sampler？
- [ ] 和 AutoDeploy 的 ADExecutor 是什么继承关系？

## 疑问

_TODO_
