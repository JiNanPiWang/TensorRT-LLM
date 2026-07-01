# 12 — export patches

> 文件: `auto_deploy/export/library/unified_attn.py`,
>       `auto_deploy/export/library/transformers_causal_mask.py`,
>       `auto_deploy/export/library/autocast_noop.py`
> 问题: torch.export 有哪些 corner case 需要 patch？

---

## 阅读笔记

_TODO_

## 关键点

- [ ] `unified_attn.py`: 怎么在 export 时替换 HF attention 为统一接口？
- [ ] `transformers_causal_mask.py`: causal mask 的 data-dependent 问题？
- [ ] `autocast_noop.py`: autocast 在 export 时的特殊处理？

## 疑问

_TODO_
