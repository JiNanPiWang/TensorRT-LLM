# 19 — KV Cache Transform

> 文件: `transform/library/kvcache.py`, `transform/library/kvcache_transformers.py`
> 方向: 把 eager attention 替换为 cached attention

---

## 核心问题

```python
# Eager attention（prefill 阶段）
q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
# K, V 用完即丢弃

# Cached attention（decode 阶段）
q_new = self.q_proj(x_new)           # 只算新 token 的 Q
k_new, v_new = self.k_proj(x_new), self.v_proj(x_new)
k_cache[slot] = k_new                # 写入 KV cache
v_cache[slot] = v_new
# 从 cache 读取所有历史 K, V
output = paged_attention(q_new, k_cache, v_cache, block_table)
```

---

## Transform 做了什么

`insert_cached_attention` 在 `cache_init` stage 中执行：

1. **找到所有 attention 模式**（SDPA、eager attention、custom op）
2. **替换为 cached attention**：输入加上 `k_cache, v_cache, block_table, slot_mapping`
3. **插入 KV cache write 节点**：在 attention 后把新 K, V 写入 cache

```python
# 图层面
insert_cached_attention(mod, cm, ...)
  → 遍历所有 GraphModule
    → 在 attention 节点处注入 cache placeholder
    → 替换 attention op 为 cached attention op
    → 添加 cache write 操作
```

---

## 关键设计：CachedSequenceInterface

```python
class CachedSequenceInterface:
    kv_cache_manager    # KVCacheManagerV2 实例
    info                # SequenceInfo（batch 状态）
    device              # GPU device
```

`CachedSequenceInterface` 是 transform 和 runtime 之间的桥梁：
- **Transform 阶段**：通过 `cm.info` 知道模型的结构（max_seq_len, vocab_size_padded, attention_type）
- **Runtime 阶段**：`cm.kv_cache_manager` 管理实际的 KV cache 分配

---

## 支持的 Attention 类型

```yaml
insert_cached_attention:        # 标准 MHA/GQA（backend: trtllm）
insert_cached_mla_attention:    # MLA（backend: flashinfer_mla）
insert_cached_ssm_attention:    # Mamba SSM（backend: triton_ssm）
insert_cached_causal_conv:      # Causal Conv（backend: cuda_causal_conv）
insert_cached_delta_rule:       # Delta Rule（backend: fla_delta）
insert_cached_gated_delta_rule: # Gated Delta Rule
insert_cached_residual_add:     # Residual Add cache
```

每种 attention 类型有不同的 cached 实现。


---
**下一节：[`20 — Sharding 自动分片`](../stage6-topics/20-sharding.md)**
