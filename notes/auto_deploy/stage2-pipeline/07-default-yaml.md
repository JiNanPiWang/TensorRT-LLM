# 07 — default.yaml：Transform 管线总配置

> 文件: `tensorrt_llm/_torch/auto_deploy/config/default.yaml`（347 行）
> 问题: 40 个 transform 按什么顺序执行？每个 stage 做什么？

---

## 10 个 Stage（执行顺序）

`Stages` enum 在 `interface.py:109` 定义。YAML 中的 `stage` 字段决定 transform 的归属：

```
Stage                   | 职责
factory                 | 构建 HF 模型（build_model）
export                  | torch.export → FX GraphModule（export_to_gm）
post_export             | 导出后低层清理：无用的 slice/add/input constraint
pattern_matcher         | 高层模式匹配，标准化图的表示
                        |   ├─ 匹配 MoE pattern → match_moe_pattern, match_dense_moe_pattern...
                        |   ├─ 匹配 attention → match_eager_attention, match_grouped_attention...
                        |   ├─ 匹配 RoPE → match_rope_pattern, match_rope_layout
                        |   ├─ 量化转换 → quantize_fp8_linear..., quantize_nvfp4...
                        |   └─ SwiGLU 匹配 → match_swiglu_pattern（要在量化之后！）
sharding                | 自动分片：detect_sharding + sharding_transform_executor
                        |   或用 apply_sharding_hints（基于 torch.ops.auto_deploy.all_reduce 标记）
weight_load             | 从 checkpoint 加载真实权重；把模型移到 GPU
                        |   strip_sharding_hints → load_weights → move_inputs_to_device
post_load_fusion  🎯    | 核心融合阶段：权重已加载，可以做算子融合
                        |   ├─ fuse_gemms, fuse_fp8_linear, fuse_nvfp4_linear
                        |   ├─ fuse_silu_mul ← 你的面试题！
                        |   ├─ fuse_rmsnorm, fuse_moe, fuse_swiglu
                        |   ├─ fuse_rope_into_trtllm_attention（attention fusion）
                        |   └─ mlir_elementwise_fusion（MLIR 自动融合，默认 disabled）
cache_init              | 替换为 cached attention + 初始化 KV cache
                        |   insert_cached_attention → initialize_cache → resize_kv_cache
visualize               | 图可视化（默认 disabled）
compile                 | 最终编译：compile_model（torch.compile + CUDA Graph）
                        |   multi_stream_moe, cleanup_identity_dtype_cast
```

**核心洞察**：`post_load_fusion` 是 fusion 的核心阶段，因为节点模式匹配（pattern_matcher 阶段）完成后，权重加载后，**融合才能拿到真实的 tensor shape 和 dtype 信息**。

---

## 为什么有些 fusion 默认 disabled？

```yaml
fuse_gemms:
  stage: post_load_fusion
  enabled: false  # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 OOM
```

这些是**尚未稳定或会导致 OOM 的融合**，默认关闭。你的工作之一可能就是修复这些问题然后默认开启。

---

## Transform 配置字段

每个 transform 的通用配置（在 `TransformConfig` Pydantic 中定义）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `stage` | 必填 | 属于哪个 stage |
| `enabled` | True | 是否启用 |
| `run_per_gm` | True | 是否对每个 sub-GraphModule 执行（False = 对整个模型） |
| `requires_clean_graph` | False | 执行前是否需要清理死节点 |
| `requires_shape_prop` | False | 执行前是否需要 shape propagation |
| `run_shape_prop` | False | 执行后是否运行 shape propagation |
| `run_graph_cleanup` | True | 执行后是否清理死节点 |
| `expect_mem_change` | False | 是否预期 GPU 内存变化 |
| `skip_on_error` | False | 出错是否跳过继续 |
| `backend` | 无 | 某些 transform 的后端选择（如 `torch-cudagraph`） |

---

## compile_model 的关键参数

```yaml
compile_model:
  stage: compile
  backend: torch-cudagraph       # torch.compile + CUDA Graph
  piecewise_enabled: true        # 启用 piecewise CUDA Graph
  piecewise_num_tokens: null     # 自动推断 capture 的 token 数
```

---

## 怎么加新 Transform

只需要在 `default.yaml` 中添加一项：
```yaml
fuse_my_new_op:
  stage: post_load_fusion
  enabled: true
  backend: triton
```

然后在代码中创建对应的 `BaseTransform` 子类并用 `TransformRegistry.register("fuse_my_new_op")` 注册。


---
**下一节：[`08 — Transform 管线驱动器`](../stage2-pipeline/08-optimizer.md)**
