# 09 — interface.py：Transform 基类与注册机制

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/interface.py`（850 行）
> 问题: 每个 transform 怎么写？注册机制是什么样的？

---

## 关键数据结构

### Stages（line 109-130）

```python
class Stages(Enum):
    FACTORY = "factory"          # 构建模型
    EXPORT = "export"            # torch.export
    POST_EXPORT = "post_export"  # 导出后清理
    PATTERN_MATCHER = "pattern_matcher"  # 模式匹配
    SHARDING = "sharding"        # 自动分片
    WEIGHT_LOAD = "weight_load"  # 权重加载
    POST_LOAD_FUSION = "post_load_fusion"  # 融合（🎯 你的主战场）
    CACHE_INIT = "cache_init"    # KV cache 初始化
    VISUALIZE = "visualize"      # 可视化
    COMPILE = "compile"          # torch.compile

    def __lt__(self, other):     # 按定义顺序排序！
        ...
```

### TransformConfig（line 146-185）

每个 transform 的 Pydantic 配置基类：

```python
class TransformConfig(BaseModel):
    stage: Stages                   # 必填：属于哪个 stage
    run_per_gm: bool = True         # True → 对每个子 GraphModule 运行 _apply
    enabled: bool = True            # False → 跳过
    requires_clean_graph: bool = False   # 执行前是否需要 DCE（消除死节点）
    requires_shape_prop: bool = False    # 执行前是否需要 shape propagation
    run_graph_cleanup: bool = True       # 执行后是否 DCE
    run_shape_prop: bool = False         # 执行后是否 shape propagation
    expect_mem_change: bool = False      # 是否预期 GPU 内存变化
    skip_on_error: bool = False          # 出错时是否继续
```

---

## BaseTransform（line 305-795）—— 所有 transform 的基类

### 核心流程：`__call__`（line 368-511）

```
1. 检查 enabled → 如果 False 直接返回
2. 前置处理（如果 requires_clean_graph/requires_shape_prop）
3. _apply(gm, cm, factory, shared_config)  ← 子类实现的核心
4. 后置处理（DCE, shape propagation）
5. 记录 transform_history + mem_history
6. 可视化（如果 debug_visualize_dir 设置）
7. 返回 mod
```

### 子类需要实现的方法

```python
def _apply(self, gm, cm, factory, shared_config) -> Tuple[GraphModule, TransformInfo]:
    """核心方法：对图做变换。gm 是 torch.fx.GraphModule"""
    raise NotImplementedError
```

如果 `run_per_gm=False`（需要操作整个模型而不是单个 subgraph），则覆盖：

```python
def _apply_to_full_model(self, model, cm, factory, shared_config):
    raise NotImplementedError
```

### _apply_per_gm_or_whole_model（line 535-555）

```python
if not self.config.run_per_gm:
    return self._apply_to_full_model(mod, cm, factory, shared_config)

# run_per_gm=True（默认）→ 遍历所有子 GraphModule
for k, graph_sub in named_graphmodules(mod):
    graph_sub, info = self._apply(graph_sub, cm, factory, shared_config)
    if k == "":     # 最外层 GraphModule
        mod = graph_sub
    else:            # 子模块
        mod.set_submodule(k, graph_sub)
```

---

## TransformRegistry（line 822-851）

```python
class TransformRegistry:
    _registry: Dict[str, Type[BaseTransform]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn: Type[BaseTransform]):
            cls._registry[name] = fn
            fn._transform_key = name   # 自动注入 transform key
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseTransform]:
        return cls._registry[name]
```

### 用法示例

```python
@TransformRegistry.register("fuse_silu_mul")
class FuseSiluMulTransform(BaseTransform):
    config: FuseSiluMulConfig   # 可扩展的配置类（继承 TransformConfig）

    def _apply(self, gm, cm, factory, shared_config):
        # 在 FX Graph 中找到 SiLU + Mul 模式并融合
        ...
        return gm, TransformInfo(num_matches=count)
```

---

## 写一个 Transform 的最小模板

```python
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    BaseTransform, TransformConfig, TransformInfo, TransformRegistry
)

class MyFusionConfig(TransformConfig):
    """可加自定义配置字段"""
    threshold: int = 128

@TransformRegistry.register("fuse_my_pattern")
class MyFusionTransform(BaseTransform):
    config: MyFusionConfig

    def _apply(self, gm, cm, factory, shared_config):
        num_matches = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.silu:
                # 找到 SiLU → 融合
                num_matches += 1
        gm.graph.eliminate_dead_code()
        return gm, TransformInfo(num_matches=num_matches)
```

---

## TransformInfo（line 256-304）

```python
@dataclass
class TransformInfo:
    skipped: bool = False       # 是否跳过
    num_matches: int = 0        # 匹配到的模式数量
    is_clean: bool = False      # 图是否"干净"（无死节点）
    has_valid_shapes: bool = False  # 是否有有效的 shape 信息
```

**`__or__`**（cleanup 后更新）：如果当前 clean 或之前 clean，就标为 clean。
**`__and__`**（apply 后合并）：两人都 clean 才 clean，取 min(num_matches)。


---
**下一节：[`10 — AutoDeploy 执行引擎`](../stage2-pipeline/10-ad-executor.md)**
