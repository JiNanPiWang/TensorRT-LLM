# 20 — Sharding：自动分片

> 文件: `transform/library/sharding.py`, `transform/library/sharding_ir.py`
> 方向: TP/EP 怎么自动检测和应用？

---

## 为什么 Sharding 在 weight_load 之前

```yaml
stage: sharding       # ← 权重还没加载到 GPU
stage: weight_load    # ← 权重从 checkpoint 加载到 GPU
```

**因为** sharding 阶段只修改图结构（插入 `all_reduce`, `all_gather` 等通信节点），不需要真实权重。真实权重在 weight_load 阶段才加载——**加载时会自动根据 sharding 的划分只加载每个 rank 需要的权重分片**。

---

## 两套 Sharding 管线

### 1. Sharding IR（`apply_sharding_hints`）—— 推荐方式

模型文件中显式标记了 `torch.ops.auto_deploy.all_reduce`：

```python
# 在模型代码中
if world_size > 1:
    x = torch.ops.auto_deploy.all_reduce(x)  # ← 显式标记
```

`apply_sharding_hints` 检测到这些标记后，自动插入对应的 NCCL 通信节点，并设置 `sharding_ir_applied = True`。

### 2. Heuristic Detection（`detect_sharding`）—— 回退方式

如果没有显式标记，`detect_sharding` 会：
- 分析图结构：找到所有 GEMM 节点
- 根据权重 shape 和 `sharding_source` 推断怎样分片
- 插入通信节点

**只有当 `apply_sharding_hints` 没有设置 `sharding_ir_applied` 时才会执行。**

---

## Sharding Dimensions

```yaml
sharding_dims: ['tp', 'ep', 'bmm']
```
- `tp` — Tensor Parallel（按 hidden_dim 切权重）
- `ep` — Expert Parallel（按 expert 维度切 MoE）
- `bmm` — Batch MatMul Parallel

---

## DistConfig

```python
@dataclass
class DistConfig:
    tp_size: int
    pp_size: int
    cp_size: int
    enable_attention_dp: bool
```

AutoDeploy 的单源真相——所有分布式配置由此管理。`Mapping`（外部 API 需要）由 `DistConfig.to_mapping()` 派生。


---
**下一节：[`21 — MoE + MLIR 融合`](../stage6-topics/21-moe-mlir.md)**
