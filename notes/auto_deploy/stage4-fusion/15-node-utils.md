# 15 — node_utils.py：图节点工具

> 文件: `tensorrt_llm/_torch/auto_deploy/utils/node_utils.py`（~1300 行）
> 问题: 怎么判断一个 node 是什么 op？FX Graph 的基本操作有哪些？

---

## 核心函数

### `is_op(node, ops)` — 最常用

```python
def is_op(node, ops) -> bool:
    """检查 node.target 是否匹配给定的 op(s)"""
```

支持单个 op 或多个 op：

```python
if is_op(node, torch.ops.aten.mul.Tensor):
    ...
if is_op(node, (torch.ops.aten.silu.default, torch.nn.functional.silu)):
    ...
```

处理了 `OpOverload` vs `OpOverloadPacket` vs `Callable` 的差异。

### `is_linear_op(node)` / `is_fake_quantized_linear_op(node)`

判断节点是否是 GEMM 或伪量化 GEMM。用在 pattern_matcher 中识别 MLP 结构。

### `extract_weight_name(node)` / `get_weight_tensor(gm, node)`

从 `get_attr` 节点提取权重名，或获取实际的 `torch.Tensor`。

### `get_op_schema(op)`

返回 `torch.FunctionSchema`，包含参数名称、类型、默认值。

### `LayerType` enum

```python
class LayerType(Enum):
    MHA = "mha"       # Multi-Head Attention
    SSM = "ssm"       # State Space Model (Mamba)
    MLP = "mlp"       # Feed-Forward
    MOE = "moe"       # Mixture of Experts
    MLA = "mla"       # Multi-head Latent Attention (DeepSeek V2/V3)
    DELTA = "delta"   # Delta Rule (linear attention)
    UNKNOWN = "unknown"
```

在 pattern_matcher 阶段使用，帮助识别每个 subgraph 是什么类型的层。

---

## FX Graph 基本操作速查

```python
# 遍历（必须用 list() 避免修改时并发问题！）
for node in list(gm.graph.nodes):
    if node.op == "placeholder":     # 图输入
    elif node.op == "get_attr":      # 权重/参数
    elif node.op == "call_function": # 函数调用（99% 的 op）
    elif node.op == "call_method":   # .contiguous(), .view() 等方法
    elif node.op == "output":        # 图输出

# 插入新节点
with gm.graph.inserting_before(target):
    new = gm.graph.call_function(op, args=(a, b))

# 替换
old.replace_all_uses_with(new)

# 修改后的清理
gm.graph.eliminate_dead_code()
gm.recompile()  # 必须！否则图状态不一致
```

---

## `meta["val"]` — FakeTensor Shape 系统

每个 `call_function` 节点的 `node.meta["val"]` 存储一个在 **meta device** 上的 `torch.Tensor`，只有 dtype + shape，没有实际数据：

```python
# 读取
val = node.meta.get("val")   # FakeTensor: shape=(4, 4096), dtype=float16
val.shape                    # → torch.Size([4, 4096])
val.dtype                    # → torch.float16

# 创建新节点时设置
fused_node.meta["val"] = torch.empty(
    ref_val.shape, dtype=ref_val.dtype, device="meta"
)
```

**关键**：`meta["val"]` 必须在创建新节点时手动设置。如果缺失，下游的 shape-dependent transform 会失败。


---
**下一节：[`16 — 图操作工具箱`](../stage4-fusion/16-graph-utils.md)**
