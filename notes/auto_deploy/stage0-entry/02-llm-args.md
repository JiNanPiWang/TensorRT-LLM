# 02 — llm_args.py：配置 Schema

> 文件: `tensorrt_llm/llmapi/llm_args.py`（~5500 行），
> `_torch/auto_deploy/llm_args.py`（~400 行）
> 问题: TRT-LLM 有多少可配参数？AutoDeploy 比标准 PyTorch 多了哪些参数？

---

## 类继承结构

```
pydantic.BaseModel
  └── StrictBaseModel
        └── BaseLlmArgs              ← PyTorch + TensorRT 共用（~1000 行字段定义）
              ├── TorchLlmArgs       ← PyTorch 后端特有字段
              │     └── LlmArgs     ← AutoDeploy 专属（`_torch/auto_deploy/llm_args.py`）
              └── TrtLlmArgs        ← TensorRT 遗留后端（不再新增特性）
```

---

## BaseLlmArgs：两个后端共用的核心参数

在 `llm_args.py:3591` 定义。关键字段：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `model` | 必填 | HF 模型路径或名称 |
| `tokenizer` | None | tokenizer 路径（None 则从 model 推断） |
| `tensor_parallel_size` | 1 | TP 分片数 |
| `pipeline_parallel_size` | 1 | PP 分片数 |
| `dtype` | "auto" | 从 HF config 推断 float16/bfloat16 |
| `max_batch_size` | 128 | 最大 batch |
| `max_seq_len` | 2048 | 最大序列长度 |
| `max_num_tokens` | 8192 | 单次 forward 最大 token 数 |
| `max_beam_width` | 1 | beam search 宽度 |
| `kv_cache_config` | 默认 | KV cache 配置（paged attention 参数） |
| `speculative_config` | None | 投机解码（MTP/Eagle） |
| `lora_config` | None | LoRA 配置 |
| `checkpoint_format` | 默认 | 权重加载格式 |
| `backend` | 自动推断 | `"pytorch"` / `"_autodeploy"` / 自动 |
| `orchestrator_type` | None | 多进程编排器（`"rpc"` / `"ray"`） |

**`extra="forbid"`**：传不在 schema 中的参数直接 ValueError，防止拼写错误。

---

## TorchLlmArgs：PyTorch 后端特有

在 `llm_args.py:4598` 定义。比 BaseLlmArgs 多出的关键字段：

| 字段 | 说明 |
|------|------|
| `cuda_graph_config` | CUDA Graph 配置（batch_sizes, max_batch_size, enable_padding） |
| `attn_backend` | Attention 后端选择：`TRTLLM` / `FlashInfer` / `FlashAttention` / `VANILLA` |
| `moe_config` | MoE 配置 |
| `multimodal_config` | 多模态配置 |
| `sampler_type` | Sampler 类型：`TorchSampler` / `TRTLLMSampler` |
| `garbage_collection_gen0_threshold` | Python GC 阈值 |
| `nvfp4_gemm_config` | NVFP4 量化 GEMM 配置 |
| `attention_dp_config` | DP Attention 负载均衡 |

---

## AutoDeploy 的 LlmArgs：你团队的专属"超集"

在 `_torch/auto_deploy/llm_args.py` 定义，**继承 `TorchLlmArgs`**。额外字段：

### 核心差异化字段

```python
# 自动分片（替代手动 TP/PP/CP）
world_size: int = 1            # GPU 数量，自动分片

# 运行时选择
runtime: Literal["demollm", "trtllm"] = "trtllm"

# 执行模式
mode: Literal["graph", "transformers"] = "graph"
#       "graph" → 完整的 FX Graph capture + transform + compile
#       "transformers" → 只用 cached attention 优化（类似 HF 推理）

# 图变换配置（核心！）
transforms: Dict[str, Dict[str, Any]] = {}
#  示例: {"fuse_silu_mul": {"enabled": true},
#         "fuse_quantized_linear": {"enabled": true}}

# 编译后端
compile_backend: str = "torch-cudagraph"
#  "torch-cudagraph" → torch.compile + CUDA Graph
#  "torch-opt"       → torch.compile without CUDA Graph
#  "torch"           → torch.compile only
```

### 禁用的父类字段

AutoDeploy **不支持**手动并行配置（因为用 `world_size` 自动分片）：

```python
@field_validator("tensor_parallel_size", "pipeline_parallel_size", ...)
def ensure_no_custom_parallel_config(cls, value, info):
    msg = "AutoDeploy only supports parallelization via the `world_size` argument."
    return _check_for_default_value_only(cls, value, info, msg)
```

如果你传 `tensor_parallel_size=4` 给 AutoDeploy → ValueError。

### 从 TorchLlmArgs → AutoDeploy LlmArgs 的关键思路

**因为** PyTorch 后端的 `TorchLlmArgs` 已经提供了模型加载、KV cache、CUDA Graph 等所有运行时参数的 schema，**所以** AutoDeploy 不需要重新发明轮子，**只需要**：
1. 加上 `transforms`（图变换链配置）
2. 加上 `compile_backend`（torch.compile 配置）
3. 用 `world_size` 替代手工 TP/PP 配置
4. 禁止用户传不支持的 TensorRT 遗留参数

---

## 参数解析的顺序

Pydantic 的 `model_validator` 按声明顺序执行。AutoDeploy `LlmArgs` 的关键执行链：

1. **`update_transforms_with_shortcuts`**：同步 `compile_backend` 之类的 shortcut 字段到 `transforms` dict 中
2. **`validate_supported_speculative_config`**：检查只支持 MTP/Eagle 投机解码
3. **`setup_hidden_state_capture`**：投机解码时自动注册 `detect_hidden_states_for_capture` transform
4. **`validate_parallel_config`**：从 `world_size` 自动推导 `_ParallelConfig`
5. **`extend_default_cuda_graph_config_to_max_batch_size`**：自动扩展 CUDA Graph 的 batch_sizes
6. **`sync_cuda_graph_batch_sizes_to_compile_config`**：同步 CUDA Graph batch_sizes 到 `compile_model` transform

---

## 总结

- **`BaseLlmArgs`** 是所有后端共用的 Pydantic schema（模型路径、并行、KV cache）
- **`TorchLlmArgs`** 加了 PyTorch 专属参数（CUDA Graph、attention 后端）
- **AutoDeploy `LlmArgs`** 继承 `TorchLlmArgs`，加了 `transforms` + `compile_backend` + `world_size`，禁用了手动并行配置
- **`extra="forbid"`** 保证参数拼写错误立即报错
