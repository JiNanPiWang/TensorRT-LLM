# 08 — optimizer.py：Transform 管线驱动器

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/optimizer.py`（144 行）
> 问题: InferenceOptimizer 怎么按顺序驱动 40+ 个 transform？

---

## 架构：简洁的工厂 + 循环

```python
class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, config, dist_config):
        self.factory = factory        # 模型工厂（管理 HF 模型加载）
        self.config = self._clean_config(config)  # 排序后的 StrictInferenceOptimizerConfig

    def __call__(self, cm, mod=None) -> nn.Module:
        # 按 stage 顺序遍历所有 transform
        for t_name, t_config in self.config.items():
            transform = self._create_transform(t_name, t_config)
            mod = transform(mod, cm, self.factory, self.shared_config, idx)
        return mod
```

只有 **144 行**，因为所有复杂逻辑都在 `BaseTransform.__call__` 里。

---

## _clean_config（line 62-75）—— 排序关键

```python
def _clean_config(self, config):
    # 1. TransformConfig → 纯字典
    nested_kwargs = {k: v.model_dump() if isinstance(v, TransformConfig) else v
                     for k, v in config.items()}
    # 2. 按 stage 排序（Factory < Export < PostExport < ... < Compile）
    keys_sorted = sorted(nested_kwargs.keys(),
                         key=lambda k: Stages(nested_kwargs[k]["stage"]))
    # 3. 重新包装为正确的 Pydantic config 类
    strict_config = {k: TransformRegistry.get_config_class(k)(**nested_kwargs[k])
                     for k in keys_sorted}
    return strict_config
```

**`Stages` enum 实现了 `__lt__`**（按定义顺序比较），所以 `sorted(key=lambda k: Stages(...))` 能正确排序。

---

## __call__（line 83-124）—— 核心管线

```python
def __call__(self, cm, mod=None):
    # 1. 尝试从 pipeline cache 恢复（避免重复运行已完成的 transform）
    if mod is None:
        restored_mod, start_idx = self._maybe_restore_from_cache(cm)
        if restored_mod is not None:
            mod = restored_mod  # 从上次中断处继续

    # 2. 按序执行 transform
    for idx, (t_name, t_config) in enumerate(self.config.items()[start_idx:], start_idx):
        transform = self._create_transform(t_name, t_config)
        mod = transform(mod, cm, self.factory, self.shared_config, idx)
        # transform.__call__ 内部：
        #   → 检查 enabled
        #   → 前置处理（clean graph, shape propagation）
        #   → 调用 _apply（子类实现的核心逻辑）
        #   → 后置处理（clean graph, 可视化, 日志）

    # 3. 清理并返回
    torch.cuda.empty_cache()
    gc.collect()
    return mod
```

---

## Pipeline Cache（line 126-138）

```python
def _maybe_restore_from_cache(self, cm):
    for t_name, t_config in reversed(list(enumerate(self._cache_key_config.items()))):
        transform_cls = TransformRegistry.get(t_name)
        if not hasattr(transform_cls, "maybe_restore"):
            continue
        transform = self._create_transform(t_name, t_config)
        restored_mod = transform.maybe_restore(cm, self.factory, self.shared_config, idx)
        if restored_mod is not None:
            return restored_mod, idx + 1  # 从下一个 transform 继续
    return None, 0
```

**目的**：如果某个 transform（如 `pipeline_cache`）之前已经缓存了优化后的模型，可以直接 restore 而不是重新跑昂贵的 export/fusion/compile。

---

## 总结

- InferenceOptimizer 是**工厂方法模式 + 责任链**的简洁实现
- 每个 transform 是独立的、可替换的模块
- 排序由 `Stages` enum 保证
- Pipeline cache 允许跳过已完成的 transform
