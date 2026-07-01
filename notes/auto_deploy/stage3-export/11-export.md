# 11 — export.py：图导出

> 文件: `tensorrt_llm/_torch/auto_deploy/export/export.py`（807 行）
> 问题: torch.export 怎么把 HF 模型变成 FX GraphModule？

---

## 为什么需要 export

PyTorch eager 模式下，每次 forward 都会重建计算图。要做图级优化（fusion, sharding），必须先把动态执行**静态化**为一棵固定的 FX 计算图。

`torch.export.export` 是 PyTorch 2.x 的图捕获 API（替代 `torch.fx.symbolic_trace`），支持：
- 控制流（`torch.cond`）
- 动态 shape（`dynamic_shapes`）
- 严格的图语义（`strict=True` 保证图中没有不支持的 Python 操作）

---

## torch_export_to_gm（line 715-807）—— 主函数

```python
def torch_export_to_gm(model, args, kwargs, *, dynamic_shapes, strict, patch_configs, ...):
    # 1. （可选）MoE 优化：trace 时只用 2 个 expert，大幅加速 export
    if num_moe_experts_for_export is not None:
        moe_reductions = _reduce_moe_experts(model, num_moe_experts_for_export, args, kwargs)

    # 2. 核心：torch.export 捕获
    egm = run_forward_for_capture(
        model, _capture_fn, args, kwargs, clone,
        patch_list=patch_list, patch_configs=patch_configs
    )
    # _capture_fn 内部调: te.export(model, args, kwargs, dynamic_shapes, strict)

    # 3. MoE 图展开（把 2 个 expert 恢复到全量）
    if moe_reductions:
        _restore_moe_experts(moe_reductions)
        _expand_moe_experts_in_graph(egm, model, moe_reductions)

    # 4. 后处理
    _add_missing_load_hooks(egm, model)       # 恢复 state_dict load hooks
    _add_load_hook_for_aliased_params(egm, model)  # 处理参数别名
    _deduplicate_params_and_buffers(egm)      # 删除重复参数
    _clean_up_device_info(egm)               # 清理 device 信息（全部用 meta）
    _clean_up_assertions_and_guards(egm)     # 删除 FX guard 节点
    _rename_nodes_with_module_hierarchy(egm) # 重命名字段，方便调试

    return egm
```

---

## MoE export 优化（line 41-95）

MoE 模型的专家数可能很大（如 Mixtral 8 个，DeepSeek 256 个）。但是每个 expert 结构相同，只是权重不同。

```python
# 优化策略：
#   trace 阶段：只保留 num_moe_experts_for_export=2 个 expert
#     → torch.export 只需追踪 2 个 expert 的子图
#     → 大幅减少 export 时间
#   export 后：展开图，复制 2 个 expert 的子图到所有 expert
#     → 每个 expert 有独立的权重引用
#     → 图中计算逻辑正确
```

关键在于 `_infer_target_pattern()` — 从两个连续 expert 的权重名（如 `experts.0.gate.weight`, `experts.1.gate.weight`）推断命名模式 `("experts.", ".gate.weight")`。

---

## Export 为什么需要 patch

`torch.export` 在严格模式下有很多限制。一些 HF transformers 和 PyTorch 的操作需要打补丁：

| Patch | 处理的问题 |
|-------|-----------|
| `unified_attn` | 统一的 attention 接口，替代 eager mode 的实现 |
| `transformers_causal_mask` | HF 的动态 causal mask 无法被 trace |
| `transformers_sdpa_mask` | HF 的 SDPA mask 格式转换 |
| `autocast_noop` | `torch.autocast` 在 export 中无意义，替换为 no-op |
| `meta_nonzero` | `torch.nonzero` 不支持 meta tensor → fake 实现 |
| `sdpa_kernel_noop` | `sdpa_kernel` context manager → no-op |
| `linear` | 替换 `nn.Linear` 为可导出的版本 |
| `tensor_meta_device` | meta device 上的 tensor 创建 |
| `torch_where` | 处理 `torch.where` 在 export 中的限制 |

---

## Run forward for capture 的特别之处

export 在 **meta device** 上运行（所有 tensor 只有 dtype + shape，没有真实数据）。这避免 export 阶段分配 GPU 内存，同时允许 shape propagation。

---

## 总结

- `torch_export_to_gm` 是 AutoDeploy 的图捕获入口，**不是简单的 `torch.export.export` 包装**
- MoE 优化：trace 2 experts → 展开到全量，大幅加速
- Export patches：16 个 patch 处理 torch.export 不支持的 corner case
- Meta device：export 阶段不需要 GPU 内存
