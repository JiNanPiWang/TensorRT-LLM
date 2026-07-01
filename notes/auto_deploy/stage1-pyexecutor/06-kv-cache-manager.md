# 06 — kv_cache_manager_v2.py：KV Cache 管理

> 文件: `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py`（~900 行），
> 底层 C++ 实现在 `bindings/internal/batch_manager/kv_cache_manager_v2_utils`
> 问题: Paged attention 的 KV cache 怎么分配/释放/复用？

---

## 为什么用 Paged Attention

传统做法是每个请求预留 `max_seq_len × hidden_size` 的连续 KV cache，浪费严重（大多数请求实际长度远小于 max_seq_len）。

Paged Attention 把 KV cache 切成固定大小的 **block**（如 `token_per_block=64`）：
- 请求需要新空间 → 分配若干 block
- 请求结束 → block 释放回池
- Block 不要求物理连续 → 灵活利用碎片

```
请求 A: [b0][b1][b3][b7]        ← 4 个 block
请求 B: [b2][b5]                ← 2 个 block
空闲:    [b4][b6][b8][b9]...   ← 等待分配
```

---

## KVCacheManagerV2 核心职责

```python
class KVCacheManagerV2:
    def allocate_generation(requests):     # decode 阶段分配 1 个 token 的 KV slot
    def allocate_context(requests):        # prefill 阶段分配所有 prompt tokens 的 KV slots
    def free(requests):                    # 请求结束 → 释放 blocks
    def get_block_tables():                # 返回所有请求的 block table → 传给 attention kernel
    def maybe_rebalance():                 # KV cache 池大小动态调整（分离式推理）
```

---

## Block Table 结构

传给 attention kernel 的关键数据结构：

```python
# 每个请求: block_table[request] = [block_idx_0, block_idx_1, ...]
# attention kernel 通过 block_table[idx] 找到物理 block
# → 计算 K, V 的物理地址 = pool_base + block_table[idx] * block_size_bytes
```

### 关键索引映射

- **sequence_length** → 逻辑 token 位置（每个请求内部）
- **slot_mapping** → 物理 slot 位置（在 pool 中）
- **block_table** → slot → block 的映射表

---

## Block 复用（Prefix Caching）

多个请求如果有相同的 prompt 前缀，可以共享 KV cache blocks：

```python
# ReuseScope: 配置复用粒度
# - NONE: 不复用
# - PREFIX: 共享前缀 blocks（但各自拥有自己的后缀）
```

通过 **hash(prefix_tokens)** 查找已有 block。

---

## 和 AutoDeploy 的关系

AutoDeploy 的 KV cache transform（Stage 6, `kvcache.py`）负责 **在编译阶段把原始 attention 替换成 cached attention**：

```python
# 原始 attention（eager 模式）
output = attention(q, k, v)

# cached attention（推理模式 —— 由 transform 插入）
k_cache[slot_mapping] = k    # 写入 KV cache
v_cache[slot_mapping] = v
output = attention_with_cache(q, k_cache, v_cache, block_table)
```

KVCacheManagerV2 只管理 **分配/释放/寻址**，不关心 attention 的计算逻辑。transform 负责把 attention 算子替换为"从 cache 读 K,V + 追加写入"的版本。

---

## 总结

- Paged Attention = 固定大小 block + 非连续分配 → 零碎片浪费
- KVCacheManagerV2 管理 block 的分配/释放/复用
- Block table 是 attention kernel 的寻址基础
- AutoDeploy 的 KV cache transform 负责在图层替换 attention 算子
