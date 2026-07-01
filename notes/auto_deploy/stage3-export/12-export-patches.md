# 12 — export patches：图导出补丁系统

> 文件: `tensorrt_llm/_torch/auto_deploy/export/interface.py`（patch 基类 + registry），
> `tensorrt_llm/_torch/auto_deploy/export/library/`（15+ 个 patch 实现）
> 问题: torch.export 不支持什么？怎么用 patch 绕过？

---

## Export Patches 是什么

`torch.export` 在 strict 模式下追踪 Python 代码时，很多操作会触发 **graph break**（图中断）。Graph break 意味着图被切成多段，无法做跨段的图优化。

Export patch 系统在 `torch.export` 执行前后**临时替换**不支持的操作，让 trace 产生完整连续的计算图。

---

## BaseExportPatch（`export/interface.py:61`）

```python
class BaseExportPatch(ABC):
    config: ExportPatchConfig  # {enabled: bool, skip_on_error: bool}
    _patch_key: str

    @abstractmethod
    def apply(self, model, name, args, kwargs): ...
    # export 时如果遇到 name 对应的操作 → 调用 apply
```

**工作流程**：
1. Export 开始前：patch 注册为 torch dispatch hook
2. `torch.export` trace 期间：遇到匹配的操作 → 调用 `apply()` 返回替代实现
3. Export 结束后：自动恢复原始操作

---

## ExportPatchConfig vs TransformConfig

| | ExportPatchConfig | TransformConfig |
|---|---|---|
| 时机 | `torch.export` trace 期间 | export 之后，对 FX Graph 操作 |
| 介入层 | Python dispatch / torch 函数层 | FX Graph IR 层 |
| 修改什么 | 替换函数实现、修改 dispatch | 修改图节点、替换 subgraph |

---

## Patch 注册 → 使用

```python
# 注册
@ExportPatchRegistry.register("my_patch")
class MyPatch(BaseExportPatch):
    def apply(self, model, name, args, kwargs):
        return alternative_implementation(*args, **kwargs)

# 导出时使用
torch_export_to_gm(model, args, kwargs, patch_configs={"my_patch": {"enabled": True}})
```

---

## 关键 Patches（`export/library/`）

| Patch | 处理的图断原因 |
|-------|---------------|
| `unified_attn.py` | HF 的 attention 实现太动态，替换为固定模式的 attention |
| `transformers_causal_mask.py` | `_prepare_4d_causal_attention_mask` 动态创建 mask → 替换为静态 mask |
| `transformers_sdpa_mask.py` | `torch.nn.functional.scaled_dot_product_attention` 的 mask 格式不兼容 |
| `autocast_noop.py` | `torch.autocast` 在 tracing 中无意义 → no-op |
| `meta_nonzero.py` | `torch.nonzero` 出来的 shape 在 meta device 上不确定 → fake 实现 |
| `sdpa_kernel_noop.py` | `torch.backends.cuda.sdp_kernel` context manager → no-op |
| `linear.py` | 替换 `nn.Linear` 为标准化的线性层 |
| `tensor_meta_device.py` | meta device 上的 tensor factory 函数 |
| `torch_where.py` | `torch.where` 在条件不确定时 graph break |
| `torch_modulelist_getitem.py` | `ModuleList[i]` 动态索引 → 静态展开 |

---

## 和 Fusion 的关系

**因为** export patches 保证 torch.export 产出完整连续的计算图，**所以** pattern_matcher 和 post_load_fusion 阶段的 transform 才能在图上进行可靠的模式匹配和融合。

如果图中有断点，跨断点的算子无法被匹配为一个 pattern，融合就会失败。Export patches 是模型融合的前置保障。
