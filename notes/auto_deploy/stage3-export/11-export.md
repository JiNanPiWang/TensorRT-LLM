# 11 — export.py

> 文件: `tensorrt_llm/_torch/auto_deploy/export/export.py`
> 行数: 807
> 问题: torch.export 导出怎么处理图中断？MoE 的 export 优化怎么做的？

---

## 阅读笔记

_TODO_

## 关键函数

| 函数 | 作用 |
|------|------|
| `torch_export_to_gm()` | 主入口：HF 模型 → FX GraphModule |
| `_infer_target_pattern()` | MoE expert 权重的命名模式推断 |
| MoE 优化 | trace 时减少 experts，导出后展开 |

## 关键点

- [ ] `torch.export.export` vs `torch_export_to_gm` 的区别？
- [ ] 哪些情况会导致 graph break？怎么 patch？
- [ ] export patches（`unified_attn`, `transformers_causal_mask`, `autocast_noop`）各自处理什么？
- [ ] meta tensor 在 export 阶段的作用？

## 疑问

_TODO_
