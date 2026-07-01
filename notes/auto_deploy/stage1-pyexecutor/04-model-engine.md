# 04 — model_engine.py：模型执行引擎

> 文件: `tensorrt_llm/_torch/pyexecutor/model_engine.py`（~5500 行）
> 问题: ModelEngine 怎么执行 forward？torch.compile 和 CUDA Graph 在哪介入？

---

## 抽象基类

```python
class ModelEngine(ABC):
    @abstractmethod
    def forward(self, scheduled_requests, resource_manager, ...):
        raise NotImplementedError

    def warmup(self, resource_manager):
        return  # 可选，子类覆盖
```

`PyTorchModelEngine` 继承 `ModelEngine`，实现 `forward` 和 `warmup`。

---

## PyTorchModelEngine.__init__（line 226-303+）

关键字段：

```python
self.model                    # HF 模型（例如 LlamaForCausalLM），已被包装
self.forward_pass_callable    # 实际调用的 forward 函数（可能被 torch.compile 包装）
self.ub_buffers              # UserBuffer：zero-copy 通信缓冲区
self.mapping                  # TP/PP 配置
self.llm_args                 # 全量配置
self.max_seq_len              # 最大序列长度
self.batch_size               # 最大 batch
```

### 模型加载逻辑

在 `__init__` 后面部分：
- 调用 `ModelLoader` 从 HF checkpoint 加载原始模型
- 应用 `model_kwargs` 覆盖模型的 config（如 `torch_dtype`）
- 包装进 `DecoderModelForCausalLM`（TRT-LLM 的模型抽象层，统一了各个 HF 模型到 TRT-LLM 的 forward 接口）
- 如果需要 torch.compile，则 `torch.compile(model)` → 赋值给 `self.forward_pass_callable`

---

## forward（line 4939-5050）—— 每次 iteration 的调用

```python
@torch.inference_mode()
@with_model_extra_attrs(lambda self: self.model.extra_attrs)
def forward(self, scheduled_requests, resource_manager, ...):
    # 1. 获取 KV cache manager（paged attention 的 block table）
    kv_cache_manager = resource_manager.get_resource_manager(...)

    # 2. 构造 attention metadata
    #    ← 包含：block_tables, slot_mappings, context_lens, sequence_lengths...
    attn_metadata = self._set_up_attn_metadata(kv_cache_manager, ...)

    # 3. 构造 spec metadata（如果有投机解码）
    if self.enable_spec_decode:
        spec_metadata = self._set_up_spec_metadata(...)
        attn_metadata.update_spec_dec_param(...)

    # 4. 准备模型输入: input_ids, position_ids, attention_mask...
    model_inputs = self._prepare_model_inputs(scheduled_requests, attn_metadata)

    # 5. 执行模型 forward（这是 GPU 计算的核心）
    #    ↓ 这里有两种路径：
    #    (a) CUDA Graph replay（如果 batch_size 命中已 capture 的）
    #    (b) eager forward（直接调用 self.forward_pass_callable）
    if cuda_graph_key is not None:
        outputs = self.cuda_graph_runner.replay(cuda_graph_key, model_inputs)
    else:
        outputs = self.forward_pass_callable(**model_inputs)

    # 6. 处理输出：logits, hidden_states, multimodal embeddings...
    return outputs
```

### `_prepare_model_inputs` 做什么

把 `scheduled_requests`（Scheduler 打包的 batch 描述）转换成实际的张量：

```python
{
    "input_ids": torch.tensor([...]),           # [batch_size, padded_seq_len]
    "position_ids": torch.tensor([...]),        # 位置编码
    "attn_metadata": AttentionMetadata(...),    # block_tables, slot_mappings, ...
    "spec_metadata": SpecMetadata(...),         # 投机解码元数据
    "multimodal_params": {...},                 # 多模态输入
    "lora_params": {...},                       # LoRA 权重索引
}
```

---

## warmup（line 4000+）—— CUDA Graph capture

```python
def warmup(self, resource_manager):
    # 1. 如果有 torch.compile，运行一次 warmup forward 触发编译
    #    torch.compile 是 JIT 编译，第一次 forward 会编译

    # 2. 遍历 cuda_graph_config.batch_sizes，逐个 capture CUDA Graph
    for bs in batch_sizes:
        # 构造 dummy 输入
        dummy_inputs = self._build_dummy_inputs(bs, ...)
        # capture CUDA Graph
        self.cuda_graph_runner.capture(bs, dummy_inputs)
```

**关键**：warmup 阶段在模型正式服务之前完成：
1. torch.compile 的 JIT 编译（第一次 forward 较慢）
2. CUDA Graph capture（每个 batch_size 一个 graph）

---

## torch.compile 怎么介入

`torch.compile` 在 `__init__` 阶段通过 `TorchCompileConfig` 配置：

```python
# 如果 llm_args.torch_compile_config.enable:
self.forward_pass_callable = torch.compile(
    self.model,
    dynamic=True,           # 支持动态 shape
    backend="inductor",     # 默认 Inductor
    options={...},          # 额外的编译选项
)
```

**关键点**：标准 PyTorch 后端的 torch.compile 是**整个模型一层直接包**，不做 graph-level 的 fusion。而 AutoDeploy 的方式是：**先用 FX Graph 做 fusion（Stage 3-4），最后给 torch.compile 的是一个已经融合过的图**。两者在 torch.compile 层面是兼容的——只是输入图的质量不同。

### AutoDeploy 的模型怎么对接 ModelEngine

在 AutoDeploy 路径中，`ADExecutor.__init__` 会：
1. 从 HF 加载模型 → 元权重（fake weights）
2. `torch.export` → FX Graph
3. Transform pipeline → 融合的图
4. `torch.compile(compiled_model)` → forward_pass_callable
5. `ADExecutor(PyExecutor)` 用编译后的模型替换原始 HF 模型

**所以 ModelEngine 不需要知道模型是原始 HF 的还是 AutoDeploy 编译后的**——它只管调用 `self.forward_pass_callable`。

---

## 总结

- ModelEngine 是模型执行的中介层：接收 Scheduler 的 batch 描述，转换成张量，调用 `forward_pass_callable`
- **torch.compile** 在 `__init__` 中包裹模型，**CUDA Graph** 在 `warmup` 中 capture
- `forward_pass_callable` 可以是：原始 HF 模型 / `torch.compile(model)` / AutoDeploy 编译后的 FX Graph
- AutoDeploy 的核心价值：给 `forward_pass_callable` 提供比原始 HF 模型更高效的编译后图


---
**下一节：[`05 — CUDA Graph`](../stage1-pyexecutor/05-cuda-graph-runner.md)**
