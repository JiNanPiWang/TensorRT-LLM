# 22 — 新模型接入 + 测试

> 文件: `auto_deploy/models/custom/`, `tests/unittest/auto_deploy/.../test_fuse_*.py`
> 方向: 怎么接新模型？怎么写 fusion 测试？

---

## 模型目录结构

```
auto_deploy/models/
├── factory.py          ← ModelFactory 入口 + ModelFactoryRegistry
├── hf.py                ← HF 模型通用加载
├── configs/             ← 模型配置（继承 PretrainedConfig）
├── patches/             ← 小修改（覆盖特定模型的 HF 行为）
└── custom/              ← 完整的 custom model 实现（新架构放在这）
```

---

## ModelFactory 注册

```python
# factory.py
class ModelFactoryRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(factory_cls):
            cls._registry[name] = factory_cls
            return factory_cls
        return decorator

# 使用：注册一个新模型
@ModelFactoryRegistry.register("MyNewModelForCausalLM")
class MyNewModelFactory(ModelFactory):
    def create_config(self, **model_kwargs):
        # 返回 PretrainedConfig 子类
    def create_model(self, config):
        # 创建模型实例
```

---

## 接新模型的流程

1. **创建 `models/custom/my_new_model.py`**
   - 定义 `MyNewModelConfig(PretrainedConfig)`
   - 定义 `MyNewModelForCausalLM(DecoderModelForCausalLM)`

2. **如果 torch.export 不支持模型的某个操作** → 添加 export patch

3. **如果模型的 attention 结构不标准** → 在 `match_eager_attention` 等 transform 中添加匹配逻辑

4. **测试**：
   - `tests/unittest/auto_deploy/test_export/test_my_new_model.py` — 验证 export 成功
   - `tests/unittest/auto_deploy/test_fuse_*.py` — 验证 fusion 正确性
   - `tests/integration/defs/...` — 端到端精度测试

---

## Fusion 测试怎么写

```python
# tests/unittest/auto_deploy/test_fuse_silu_mul.py
class TestFuseSiluMul:
    def test_fuse_silu_mul_basic(self):
        # 1. 构造输入 FX Graph
        gm = _build_test_graph()  # 包含 silu(narrow) * narrow 的模式

        # 2. 运行 transform
        transform = FuseSiluMul.from_kwargs(enabled=True, backend="flashinfer")
        gm, info = transform._apply(gm, ...)

        # 3. 验证
        assert info.num_matches == 1
        # 检查图中有 silu_and_mul 节点，没有 silu + mul 分离的节点
        assert _count_ops(gm, torch.ops.auto_deploy.flashinfer_silu_and_mul.default) == 1
        assert _count_ops(gm, torch.ops.aten.silu.default) == 0

    def test_fuse_silu_mul_narrow_and_split(self):
        # 测试两种 narrow 形式（torch.narrow / split + getitem）
        ...

    def test_fuse_silu_mul_no_match_when_different_parents(self):
        # 负面测试：两个 narrow 来自不同 parent → 不融合
        ...
```

### 验证策略

| 层次 | 方式 | 描述 |
|------|------|------|
| 图结构 | `_count_ops(gm, op)` | 检查融合后图中是否存在目标 op |
| 数值对拍 | `torch.allclose(original_output, fused_output)` | 确保融合不改变数值 |
| Shape 验证 | `node.meta["val"].shape` | 确保 shape 信息传播正确 |
| 端到端 | integration test with real model | 完整推理 + 精度对比 |
