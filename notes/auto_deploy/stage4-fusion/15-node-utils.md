# 15 — node_utils.py

> 文件: `tensorrt_llm/_torch/auto_deploy/utils/node_utils.py`
> 行数: 1896
> 问题: 有哪些节点判断/工具函数？写 transform 时怎么用？

---

## 阅读笔记

_TODO_（浏览即可）

## 核心函数

| 函数 | 作用 |
|------|------|
| `is_op(node, op)` | 判断节点是不是某个 op |
| `is_linear_op(node)` | 判断是不是 GEMM |
| `is_fake_quantized_linear_op(node)` | 判断是不是伪量化 GEMM |
| `extract_weight_name(node)` | 获取参数名 |
| `get_op_schema(node)` | 获取 op schema |
| `get_weight_tensor(gm, node)` | 获取实际权重 tensor |

## 其他

- `LayerType` enum: MHA / SSM / MLP / MOE / MLA
- `WeightBiasInfoCache`

## 疑问

_TODO_
