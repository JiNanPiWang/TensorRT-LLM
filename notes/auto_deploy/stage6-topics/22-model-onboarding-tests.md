# 22 — 新模型接入 + 测试

> 文件: `auto_deploy/models/custom/`（模型适配），
>       `tests/unittest/auto_deploy/.../test_fuse_*.py`（fusion 测试）
> 方向: 怎么接新模型 + 怎么写 fusion 测试

---

## 阅读笔记

_TODO_

## 模型目录

```
auto_deploy/models/
├── hf.py        ← HF 模型通用处理
├── factory.py   ← ModelFactory 入口
├── patches/     ← 模型特定小修改
└── custom/      ← 完整 custom model 实现
```

## 测试关键点

- [ ] 怎么构造输入 FX Graph？
- [ ] 怎么验证融合正确性？（数值对拍 vs 图结构检查）
- [ ] 哪些测试需要 GPU？

## 疑问

_TODO_
