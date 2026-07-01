# TensorRT-LLM PyTorch Backend 入门与阅读指南

> 写给有 AI 编译器 / vLLM / torch.compile 背景的新人。
> 按因果逻辑组织：先理解"系统为什么这样设计"，再看"代码在哪里、怎么读"。

---

## 第一章：TRT-LLM 是什么

TensorRT-LLM 是 NVIDIA 的 LLM 推理框架。和 vLLM 一样，它解决的核心问题是：**给定一个 HuggingFace 模型，如何在 NVIDIA GPU 上最快地做推理**。

和 vLLM 的关键区别：

| | vLLM | TRT-LLM |
|---|---|---|
| 主力后端 | PyTorch eager + CUDA Graph | **两条路径**: PyTorch eager + **AutoDeploy（图编译）** |
| 图优化 | 几乎没有（依赖 torch.compile） | AutoDeploy 有完整的 FX Graph → 融合 → 编译 pipeline |
| 调度 | 简单 continuous batching | C++ BatchManager（更复杂的分页/抢占/优先级） |
| 量化 | AWQ/GPTQ/FP8 | FP8/NVFP4/INT4/FineGrainedFP8/MXFP4 — 更丰富的量化方案 |
| 分布式 | Ray/TP | MPI/NCCL + TP/PP/EP/CP — 更底层的并行策略 |
| 分离式推理 | 社区支持 | 原生 DSA（prefill/decode 跨 GPU 拆分 + NIXL KV 传输） |

TRT-LLM 有三个后端（见 `AGENTS.md:52-56`）：

```
Backend     | 状态   | 入口                     | 说明
PyTorch     | 默认   | TorchLlmArgs → PyExecutor | 主力，新功能都在这
AutoDeploy  | Beta   | ad_executor.py → graph transforms | PyTorch 的图编译增强版
TensorRT    | 遗留   | TrtLlmArgs → TensorRT Engine | 不再新增功能
```

**你入职后只碰 PyTorch + AutoDeploy，TensorRT 后端跟你无关。**

---

## 第二章：PyTorch Backend 架构——一条请求的完整旅程

### 2.1 从用户代码到 GPU 执行

```
用户代码: LLM("meta-llama/Llama-3.1-8B").generate("Hello")
    │
    ▼
llmapi/llm.py: LLM 类                     ← 用户 API 入口
    │ 解析参数 → TorchLlmArgs (Pydantic)
    ▼
llmapi/llm_args.py: TorchLlmArgs          ← 所有配置的 Schema（3000+ 行）
    │ backend="pytorch" → 走 PyExecutor
    ▼
_torch/pyexecutor/py_executor.py: PyExecutor  ← 运行时引擎
    │
    ├─ Scheduler: 把请求打包成 batch（continuous batching）
    ├─ ModelEngine: 执行 model.forward()
    ├─ KVCacheManagerV2: 管理 KV cache（paged attention）
    ├─ CudaGraphRunner: CUDA Graph capture/replay
    └─ Sampler: 从 logits 采样 token
    │
    ▼
_torch/modules/: 模型层实现                ← attention, MLP, MoE, RMSNorm...
    │ 调用底层 kernel
    ▼
_torch/attention_backend/: Attention 后端  ← FlashInfer, TRTLLM, FlashAttention
_torch/custom_ops/: 自定义 CUDA/Triton op
    │
    ▼
GPU 执行
```

### 2.2 关键文件速查

| 你在看什么 | 文件 | 行数 |
|-----------|------|------|
| 用户入口 | `llmapi/llm.py` | ~2000 |
| 配置 Schema | `llmapi/llm_args.py` | ~5500 |
| 运行时引擎 | `_torch/pyexecutor/py_executor.py` | ~1800 |
| 模型执行 | `_torch/pyexecutor/model_engine.py` | ~2500 |
| KV Cache 管理 | `_torch/pyexecutor/kv_cache_manager_v2.py` | ~900 |
| CUDA Graph | `_torch/pyexecutor/cuda_graph_runner.py` | ~600 |

### 2.3 PyExecutor 做了什么

`PyExecutor`（`py_executor.py:1800行`）是一个事件循环：

```
while True:
    requests = scheduler.schedule()          # 1. 调度：选哪些请求组成 batch
    model_engine.forward(requests)           # 2. 执行：跑模型 forward
    sampler.sample(logits, requests)         # 3. 采样：从 logits 选 token
    kv_cache_manager.update(requests)        # 4. 更新 KV cache
    send_responses(requests)                 # 5. 返回结果给用户
```

这个循环和 vLLM 的 scheduler + model runner 几乎一样。关键区别在于第 2 步——`model_engine.forward()` 里面发生了什么。

---

## 第三章：为什么需要 AutoDeploy——从 Eager 到 Graph

### 3.1 问题：Eager 模式的瓶颈

在 vLLM 或 TRT-LLM 的 **eager 模式** 中，`model.forward()` 走的是纯 PyTorch：

```python
# 这是 HuggingFace 模型的原始 forward（简化）
def forward(self, hidden_states):
    qkv = self.qkv_proj(hidden_states)   # 一次 GEMM
    q, k, v = qkv.split(...)             # split
    q, k = self.rotary_emb(q, k)         # RoPE
    attn_out = self.attention(q, k, v)   # Attention
    gate, up = self.gate_proj(x), self.up_proj(x)  # 两次 GEMM
    hidden = F.silu(gate) * up           # SiLU + Mul
    out = self.down_proj(hidden)         # 又一次 GEMM
    return out
```

问题：
1. **Kernel launch 开销大**：每个 `torch.matmul`、`F.silu` 都是一次独立的 kernel launch。DGX H100 上一个 decoder layer 可能有 20+ 次 launch。
2. **没有融合**：`gate_proj` 和 `up_proj` 共享同一个输入 `x`，但它们是两个独立的 GEMM。如果能融合成一个 GEMM，可以减少一次 kernel launch + 一次内存 I/O。
3. **算子选择不灵活**：无法自动把 SiLU+Mul 替换成融合 kernel（FlashInfer 的 `silu_and_mul`）。

### 3.2 解决：auto_deploy 的图编译 pipeline

AutoDeploy 的核心思路和你的编译器背景完全对应：

```
HuggingFace 模型 → torch.export（导出 FX Graph）
    → Pattern Matcher（识别标准 pattern：attention、RoPE、SwiGLU...）
    → Quantization（插入量化/dequant）
    → Sharding（插入 all-reduce、TP 切分）
    → Weight Loading（加载真实权重到 GPU）
    → Fusion（算子融合：GEMM fusion、SiLU+Mul fusion、RMSNorm fusion...）
    → KV Cache Insertion（替换 attention 为带 cache 的版本）
    → Compile（torch.compile + CUDA Graph）
    → 推理
```

这和你熟悉的 LLVM pass pipeline 很像，只不过操作的不是 IR 而是 **FX Graph**（PyTorch 的计算图表示）。

### 3.3 AutoDeploy 和 PyExecutor 的关系

AutoDeploy **不是**替换 PyExecutor。它是 `PyExecutor` 的一个子类（`ad_executor.py`）：

```python
# ad_executor.py:35
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

class ADExecutor(PyExecutor):  # 继承 PyExecutor
    ...
```

区别在于：
- **标准的 PyExecutor**：直接用 HuggingFace 模型的 `forward()`，可以加 `torch.compile`
- **AutoDeploy 的 ADExecutor**：先把模型跑一遍完整的图变换 pipeline（`InferenceOptimizer`），然后交给 PyExecutor 的调度/执行循环

所以 AutoDeploy 里你写的 fusion transform，最终会被 PyExecutor 的 `ModelEngine.forward()` 调用。这两个模块是上下游关系。

---

## 第四章：你的工作会落在哪里——因果链

基于代码库结构和团队分工：

**因为** TRT-LLM 的 PyTorch 后端有两层：运行时（PyExecutor）和图优化（AutoDeploy），
**所以** 需要一个从图优化到运行时的完整通路。

**因为** AutoDeploy 目前有 50+ 个 fusion/transform，但 placement 策略、图中断处理、新模型的 pattern matching 规则仍在快速演进，
**所以** 需要有人专门负责图变换层的开发。

**因为** PyExecutor 这边在做 torch.compile 集成、piecewise CUDA graph、DSA（分离式推理），
**所以** 图变换产出的图必须兼容这些运行时特性。

**你的角色**：在图变换层（AutoDeploy transform library）做开发，向下兼容 PyExecutor 运行时的需求。具体来说：

1. **写新的 fusion transform**（如 `fuse_silu_mul.py` 那样的模式）
2. **优化 fusion placement 策略**（融合节点放在图中的哪个位置）
3. **处理 torch.export 的图中间断**（新模型导出时的 corner case）
4. **配合 PyExecutor 的 compile/CUDA Graph 需求**（确保融合后的图能被 torch.compile + captured）

---

## 第五章：阅读计划——按逻辑依赖组织

阅读顺序按**依赖链**排列，而不是按"重要性"。每个阶段有明确的"因为...所以..."。

### 阶段 0：宏观入口（先跑通一条路）

**因为** 你需要知道用户怎么用 TRT-LLM 以及请求怎么到达你的代码，
**所以** 先从最外层的 API 看起。

| # | 文件 | 行数 | 看什么 |
|---|------|------|--------|
| 0.1 | `llmapi/llm.py` | ~2000 | `LLM.__init__` 和 `LLM.generate` — 用户入口，怎么创建 executor、怎么发请求 |
| 0.2 | `llmapi/llm_args.py` | ~5500 | `TorchLlmArgs` 类 — **浏览即可**，重点看 `backend` 字段、`cuda_graph_config`、`torch_compile_config` |
| 0.3 | `_torch/pyexecutor/py_executor.py` | ~1800 | `PyExecutor.__init__` 和主循环 — 理解 Scheduler → ModelEngine → Sampler 的流程 |

**读完这三篇你应该能回答**：一条 generate 请求从 Python API 到 GPU 执行经过了哪些模块。

### 阶段 1：PyExecutor 运行时核心（理解下游）

**因为** AutoDeploy 产出的图最终被 PyExecutor 的 ModelEngine 执行，
**所以** 你需要理解 ModelEngine 怎么跑 forward、怎么处理 CUDA Graph。

| # | 文件 | 行数 | 看什么 |
|---|------|------|--------|
| 1.1 | `_torch/pyexecutor/model_engine.py` | ~2500 | `PyTorchModelEngine.forward` — 模型的 forward 是怎么被调用的，torch.compile 在哪介入 |
| 1.2 | `_torch/pyexecutor/cuda_graph_runner.py` | ~600 | piecewise CUDA graph capture/replay 的机制 |
| 1.3 | `_torch/pyexecutor/kv_cache_manager_v2.py` | ~900 | KV cache 的 paged attention 管理 |
| 1.4 | `_torch/pyexecutor/scheduler.py` | ~300 | 调度逻辑（怎么选请求组 batch） |

**读完这四篇你应该能回答**：ModelEngine 怎么执行 forward、CUDA Graph 什么时候 capture、KV cache 怎么分配。

### 阶段 2：AutoDeploy pipeline 架构（理解上游）

**因为** 你主要工作在 AutoDeploy 的图变换层，
**所以** 需要理解整个 pipeline 的入口、顺序和 transform 抽象。

| # | 文件 | 行数 | 看什么 |
|---|------|------|--------|
| 2.1 | `auto_deploy/config/default.yaml` | 346 | **最重要**：40 个 transform 的完整注册表，按 stage 排列。每行都是一个 pass。 |
| 2.2 | `auto_deploy/transform/optimizer.py` | 144 | `InferenceOptimizer.__call__` — 14 行核心循环，遍历所有 transform |
| 2.3 | `auto_deploy/transform/interface.py` | 850 | `BaseTransform` 基类 + `TransformRegistry` 注册机制 — 怎么写一个新 transform |
| 2.4 | `auto_deploy/shim/ad_executor.py` | ~500 | ADExecutor 怎么继承 PyExecutor，怎么在 `__init__` 里跑完整的图优化 pipeline |

**读完这四篇你应该能回答**：用户用 `backend="auto_deploy"` 时，模型从 HuggingFace 到可推理状态经过了哪些 transform。

### 阶段 3：torch.export 导出（理解图的来源）

**因为** AutoDeploy 操作的是 torch.export 导出的 FX Graph，
**所以** 需要理解导出过程、图中间断的原因和处理方式。

| # | 文件 | 行数 | 看什么 |
|---|------|------|--------|
| 3.1 | `auto_deploy/export/export.py` | 807 | `torch_export_to_gm()` — 封装 torch.export，处理各种 corner case |
| 3.2 | `auto_deploy/export/library/unified_attn.py` | ~150 | 在 export 时替换 HuggingFace attention 为统一接口 |
| 3.3 | `auto_deploy/export/library/transformers_causal_mask.py` | ~60 | causal mask 的 patch |

**读完这三篇你应该能回答**：什么情况会导致 torch.export 图中间断，AutoDeploy 怎么 patch 这些问题。

### 阶段 4：融合核心（你的主战场）

**因为** 图已经导出、标准化，现在要开始做融合优化，
**所以** 这是你写代码的直接参考。

| # | 文件 | 行数 | 看什么 |
|---|------|------|--------|
| 4.1 | `auto_deploy/transform/library/fusion.py` | 693 | **必读**：GEMM 融合。`_insert_fused_gemm`、融合条件检查、placement 策略 |
| 4.2 | `auto_deploy/transform/library/fuse_silu_mul.py` | 406 | **必读**：完整 fusion transform 模板。照着这个模式写新 fusion |
| 4.3 | `auto_deploy/utils/node_utils.py` | 1896 | 浏览：`is_op`、`is_linear_op`、`extract_weight_name` 等工具函数 |
| 4.4 | `auto_deploy/utils/_graph.py` | 851 | 浏览：`canonicalize_graph`、`eliminate_dead_code`、`run_shape_prop` |
| 4.5 | `auto_deploy/utils/pattern_matcher.py` | 532 | 浏览：子图匹配框架 |

**读完这五篇你应该能回答**：怎么写一个新的 fusion transform，融合的 placement 怎么选，融合前要检查哪些条件。

### 阶段 5：编译后端（连接融合和运行时）

**因为** 融合后的图需要被 torch.compile 和 CUDA Graph 编译才能高效执行，
**所以** 需要理解 compile stage 做了什么。

| # | 文件 | 行数 | 看什么 |
|---|------|------|--------|
| 5.1 | `auto_deploy/compile/compiler.py` | 57 | `CompilerBackend` 抽象基类 |
| 5.2 | `auto_deploy/compile/backends/torch_compile.py` | 32 | `torch.compile(model, dynamic=True)` |
| 5.3 | `auto_deploy/compile/backends/torch_opt.py` | ~40 | torch.compile + CUDA Graph 混合 |

### 阶段 6：专题深入（按需选读）

| 方向 | 文件 | 说明 |
|------|------|------|
| Attention 融合 | `auto_deploy/transform/library/fuse_rope_into_trtllm_attention.py` | RoPE 融入 attention |
| KV Cache 插入 | `auto_deploy/transform/library/kvcache.py` | 替换 attention 为 cached 版本 |
| MoE 融合 | `auto_deploy/transform/library/fused_moe.py` | 多 expert 权重合并 |
| 量化融合 | `auto_deploy/transform/library/quantization.py` | FP8/NVFP4 量化插入 |
| Sharding | `auto_deploy/transform/library/sharding.py` | TP/EP 切分 |
| MLIR 融合 | `auto_deploy/transform/library/mlir_elementwise_fusion.py` | FX→MLIR→Triton |
| 新模型接入 | `auto_deploy/models/custom/` | 模型适配参考 |
| Attention 后端 | `_torch/attention_backend/` | Attention kernel 实现 |
| Custom Ops | `_torch/custom_ops/` | 手写 CUDA/Triton 算子 |
| 测试 | `tests/unittest/auto_deploy/.../test_fuse_silu_mul.py` | 怎么写 fusion 测试 |

---

## 附录 A：AutoDeploy Pipeline 全景（default.yaml 简化版）

```
Stage: factory
  build_model                    → HuggingFace → nn.Module

Stage: export
  export_to_gm                   → torch.export → FX GraphModule

Stage: post_export
  cleanup_noop_*                 → 清理无用的 slice/add/constraint

Stage: pattern_matcher
  match_moe_pattern              → 识别 MoE 结构
  match_eager_attention          → 识别 attention pattern
  match_rope_pattern             → 识别 RoPE
  match_rmsnorm_pattern          → 识别 RMSNorm
  match_swiglu_pattern           → 识别 SwiGLU
  quantize_fp8/nvfp4_*           → 量化变换

Stage: sharding
  apply_sharding_hints           → 插入 all-reduce、TP 切分
  detect_sharding                → 自动检测可切分区域

Stage: weight_load
  load_weights                   → 加载真实权重到 GPU
  move_inputs_to_device          → 移动输入到 GPU

Stage: post_load_fusion     ← 🎯 你的核心方向
  fuse_gemms                     → GEMM 融合（多个 linear 合并）
  fuse_gemms_mixed_children      → 宽松版 GEMM 融合
  fuse_silu_mul                  → SiLU+Mul → silu_and_mul
  fuse_rmsnorm                   → RMSNorm 融合
  fuse_moe                       → MoE 专家融合
  fuse_rope_into_trtllm_attn     → RoPE 融入 attention
  fuse_swiglu/fuse_add_rms_norm  → 其他融合
  mlir_elementwise_fusion        → MLIR 自动融合

Stage: cache_init
  insert_cached_attention        → 替换为 KV cache 版本
  initialize_cache               → 初始化 KV cache 内存池

Stage: compile
  compile_model                  → torch.compile + CUDA Graph
```

## 附录 B：值得关注的技术点

| 技术点 | 对应代码 |
|--------|---------|
| torch.compile / Dynamo 编译缓存 | `auto_deploy/compile/backends/torch_compile.py` |
| 图中间断（graph break）的场景与处理 | `auto_deploy/export/export.py` |
| 算子融合的约束条件 | `auto_deploy/transform/library/fusion.py:check_same_children()` |
| 融合节点的 placement 策略 | `auto_deploy/transform/library/fusion.py:147`, `inserting_before` vs `inserting_after` |
| Transform pipeline 的拓扑排序 | `auto_deploy/transform/optimizer.py` 按 stage 排序 |
| KV Cache 的插入与初始化 | `auto_deploy/transform/library/kvcache.py` |
| piecewise CUDA graph | `_torch/pyexecutor/cuda_graph_runner.py` |

## 附录 C：关键 API 速查

```python
# 节点判断
is_op(node, torch.ops.aten.silu.default)    # 是不是 silu
is_linear_op(node)                           # 是不是 linear/GEMM
extract_weight_name(node)                    # 获取权重参数名

# 图操作
with gm.graph.inserting_before(node):        # 在 node 前插入
    new_node = gm.graph.call_function(op, args=(...))
old_node.replace_all_uses_with(new_node)     # 替换所有 consumer
gm.graph.eliminate_dead_code()               # 清理死节点
gm.recompile()                               # 同步 Python forward

# Shape 传播
run_shape_prop(gm)                           # FakeTensor shape 推断
```
