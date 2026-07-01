# 16 — _graph.py + pattern_matcher.py

> 文件: `auto_deploy/utils/_graph.py` (851行), `auto_deploy/utils/pattern_matcher.py` (532行)
> 问题: 图操作工具箱和 pattern matching 框架？

---

## 阅读笔记

_TODO_（浏览即可）

## _graph.py 关键函数

| 函数 | 作用 |
|------|------|
| `canonicalize_graph()` | 图规范化 |
| `eliminate_dead_code()` | 删除无 consumer 节点 |
| `delete_all_unused_submodules()` | 清理未使用子模块 |
| `run_shape_prop()` | FakeTensor shape 传播 |
| `lift_to_meta()` | 参数/buffer 提升到 meta |
| `add_graph_input()` | 动态添加图输入 |

## pattern_matcher.py 关键点

- [ ] SubgraphMatcher 的匹配算法？
- [ ] 什么场景用 pattern_matcher vs 手写遍历？

## 疑问

_TODO_
