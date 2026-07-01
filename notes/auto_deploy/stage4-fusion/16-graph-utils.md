# 16 — _graph.py + pattern_matcher.py：图操作工具箱

> 文件: `tensorrt_llm/_torch/auto_deploy/utils/_graph.py`（851 行），
> `tensorrt_llm/_torch/auto_deploy/utils/pattern_matcher.py`（532 行）
> 问题: FX Graph 的底层操作和 pattern matching 框架是什么？

---

## _graph.py 关键函数

### `canonicalize_graph(gm)` — 图规范化

消除冗余操作：
- 合并连续的 `get_attr` → 同一个引用
- 消除 identity 操作
- 规范化节点名

在 `cleanup_*` 系列 transform 中调用。

### `eliminate_dead_code(gm)` — 消除死节点

删除没有 consumer 的节点（被 replace_all_uses_with 后遗弃的旧节点）。每个 fusion transform 的标准后处理步骤。

### `run_shape_prop(gm)` — Shape Propagation

用 FakeTensor 传播 shape 信息到每个节点的 `meta["val"]`。在 `cast_to_fp8`、`_to_copy` 等会改变 shape/dtype 的节点之后需要运行。

### `lift_to_meta(gm)` — 参数提升到 meta

把图中所有 `get_attr` 节点指向的 tensor 移到 meta device（保留 dtype + shape，丢弃数据）。因为 export 阶段在 meta device 上运行。

### `add_graph_input(gm, name, val=None)` — 动态添加图输入

在图中添加新的 placeholder 节点。KV cache insert 阶段用这个函数把 `kv_cache`、`block_tables` 等运行时输入注入图中。

### `named_graphmodules(mod)` — 遍历所有子 GraphModule

返回 `(name, submodule)` 的生成器。`BaseTransform._apply_per_gm_or_whole_model` 调用这个函数。

### `placeholders_on_meta(gm)` — 设置 placeholder meta

确保所有 placeholder 节点的 `meta["val"]` 都正确设置为 FakeTensor。

---

## pattern_matcher.py — SubgraphMatcher

### 什么时候用

当 pattern 是固定的多节点结构（如 attention block）时，`SubgraphMatcher` 比手写遍历更清晰。

### 使用方式

```python
from tensorrt_llm._torch.auto_deploy.utils.pattern_matcher import SubgraphMatcher

# 定义 pattern
pattern_gm = build_pattern_graph()  # 构造一个包含目标 pattern 的 GraphModule

# 匹配
matcher = SubgraphMatcher(pattern_gm)
for match in matcher.match(target_gm):
    # match 是 dict: pattern_node → target_node
    qkv_node = match[qkv_pattern_node]
    ...
```

### SubgraphMatcher vs 手写遍历

| | SubgraphMatcher | 手写遍历（如 fuse_silu_mul） |
|---|---|---|
| 优点 | 声明式、可读性高 | 灵活、性能好 |
| 缺点 | 匹配开销大、复杂 pattern 难表达 | 需要仔细处理图的遍历 |
| 适用 | 固定的多节点结构（attention, MoE） | 少量节点的简单融合（silu_mul, rmsnorm） |

**大多数 fusion 用手写遍历**——因为 pattern 通常是 2-4 个节点，遍历开销小且更灵活。

---

## graph_writer.py — 图转储

```python
# 环境变量 AD_DUMP_GRAPHS_DIR 控制
graph_writer.dump_graph(mod, t_name="fuse_silu_mul", stage="post_load_fusion")
```

在 `BaseTransform.__call__` 的 line 508 调用，每个 transform 执行后自动 dump 图。用于 debug 时查看每个 transform 前后的图变化。


---
**下一节：[`17 — 编译后端`](../stage5-compile/17-torch-compile-backend.md)**
