# 18 — Attention 融合

> 文件: `transform/library/fuse_rope_into_trtllm_attention.py`,
> `transform/library/fuse_rope_mla.py`,
> `transform/library/fuse_rope_into_trtllm_mla.py`
> 方向: 把 RoPE 融入 attention kernel，避免额外的旋转计算

---

## 为什么需要 RoPE fusion

```python
# 原始（分离的）
q, k, v = qkv_proj(x)
q_rope = apply_rotary_pos_emb(q, cos, sin)   # ← 单独一次操作
k_rope = apply_rotary_pos_emb(k, cos, sin)   # ← 又一次
output = attention(q_rope, k_rope, v)

# 融合后
output = attention_with_rope(qkv_proj(x), cos, sin)
# RoPE 旋转在 attention kernel 内部完成
```

**节省**：2 次 memory round-trip（q 和 k 不需要先写回再读入 attention）。

---

## Placement 策略

RoPE fusion 的 placement 和 `fuse_silu_mul` 类似：

```python
with gm.graph.inserting_before(attention_node):
    q_node = ...  # 找到 qkv_proj 的输出
    fused = gm.graph.call_function(
        torch.ops.auto_deploy.trtllm_attention_with_rope,
        args=(q_node, k_node, v_node, cos, sin),
    )
attention_node.replace_all_uses_with(fused)
```

**注意**：必须在 attention 节点**之前**插入，因为 RoPE 操作通常在 attention 之前，不破坏拓扑顺序。

---

## MLA（Multi-head Latent Attention）

DeepSeek V2/V3 使用 MLA，其中 KV 是低秩压缩的（latent 表示）。RoPE 融合到 MLA 比 MHA 复杂：
- MLA 的 K 和 V 是从 `kv_latent` 通过 `kv_a_proj_with_mqa` 扩展的
- RoPE 只应用到 K 的特定部分（不是全部 head dim）
- 融合时需要正确处理 MLA 的 latent → expanded → rope → attention 的链条

详细内容在 `ATTENTION_DEVELOPER_GUIDE.md`。


---
**下一节：[`19 — KV Cache Transform`](../stage6-topics/19-kvcache-transform.md)**
