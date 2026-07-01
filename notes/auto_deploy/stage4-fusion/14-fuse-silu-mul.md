# 14 — fuse_silu_mul.py：完整的 Fusion Transform 模板

> 文件: `tensorrt_llm/_torch/auto_deploy/transform/library/fuse_silu_mul.py`（406 行）
> 问题: 从头到尾写一个 fusion transform 的标准流程是什么？

---

## 融合前后

```
Before: silu(narrow(x, 0, N)) * narrow(x, N, N)
         └── silu(narrow_0) ──┐
                              ├── mul
         └── narrow_N ────────┘

After:  silu_and_mul(x)
```

从 4 个 op（2×narrow + silu + mul）→ 1 个 fused op。

---

## 完整代码结构（模板模式）

```python
# ① 自定义 Config（继承 TransformConfig）
class FuseSiluMulConfig(TransformConfig):
    backend: str = Field(default="flashinfer")  # 额外字段

# ② 注册到 TransformRegistry
@TransformRegistry.register("fuse_silu_mul")
class FuseSiluMul(BaseTransform):
    config: FuseSiluMulConfig

    @classmethod
    def get_config_class(cls):
        return FuseSiluMulConfig  # 告诉 BaseTransform 用自定义的 config 类

    # ③ 核心方法
    def _apply(self, gm, cm, factory, shared_config):
        if not self.config.enabled:
            return gm, TransformInfo(skipped=True)

        cnt = 0
        for node in list(gm.graph.nodes):       # ← list() 避免在遍历时修改图
            if not is_op(node, torch.ops.aten.mul.Tensor):
                continue                         # ← 快速跳过不匹配的节点

            result = self._try_fuse_mul(node)    # ← 尝试匹配 pattern
            if result is None:
                continue

            fused_parent, half_size = result

            # ④ placement：在原始节点之前插入新节点
            with gm.graph.inserting_before(node):
                fused_node = gm.graph.call_function(
                    torch.ops.auto_deploy.flashinfer_silu_and_mul.default,
                    args=(fused_parent,),
                )
                # 传播 shape metadata（必须！后续 transform 依赖 meta["val"]）
                fused_node.meta["val"] = torch.empty(ref_val.shape, dtype=ref_val.dtype,
                                                     device="meta")

            # ⑤ 替换所有引用
            node.replace_all_uses_with(fused_node)
            cnt += 1

        # ⑥ 消除死代码 + 重编译
        if cnt > 0:
            gm.graph.eliminate_dead_code()
            gm.recompile()

        return gm, TransformInfo(num_matches=cnt)
```

---

## Pattern 匹配：`_try_fuse_mul → _match_silu_narrow_mul`

```python
# 从 mul 节点出发，尝试两种顺序：
for silu_candidate, up_candidate in [(left, right), (right, left)]:
    # 检查 silu_candidate 是否是 silu(narrow(x, 0, N))
    # 检查 up_candidate 是否是 narrow(x, N, N)
    # N == half_size，且 x 相同
    # parent.shape[-1] == 2 * N（确保 narrow 完全覆盖 parent）
```

**关键安全检查**：`parent.shape[-1] == 2 * half_size`

**因为** silu_and_mul 的优化 kernel 假设输入按 axis=-1 前半是 gate、后半是 up，**所以** 必须验证两个 narrow 完全覆盖 parent 的最后一维。如果 parent 有其他 consumer 用了中间的部分 → 不融合。

---

## `_get_narrow_info`——处理两种 narrow 形式

```python
# 形式 1: torch.narrow(input, dim=-1, start=0, length=N)
if is_op(node, torch.narrow):
    return parent, offset, length

# 形式 2: getitem(split_output(input), idx)  ← GEMM 融合产生
if is_op(node, operator.getitem):
    # 从 split_node 的 meta["val"].shape[-1] 推断 size
    # 从 idx 推断 offset（sum of previous sizes）
```

**为什么有两种形式？**`fuse_gemms` 融合后产生 split，`fuse_gemms_mixed_children` 产生 narrow。`_get_narrow_info` 统一处理。

---

## `_strip_contiguous`——穿透 `.contiguous()`

当 `allow_not_contiguous=False` 时，fuse_gemms 会在 view 操作后面插 `.contiguous()` 调用。

```python
def _strip_contiguous(node):
    while isinstance(node, Node):
        if node.target == "contiguous":   # 跳过 .contiguous()
            node = node.args[0]
        else:
            break
    return node
```

---

## FP8 量化折叠：`_try_fuse_fp8_quant`

当 `backend="trtllm"` 且下游唯一的 consumer 是 FP8 linear 时：

```python
# 原来：
#   silu_and_mul(x) → fp8_linear(hidden, weight, bias, input_scale)
#   scaleMatrixPerTensorVec(hidden, scale)  ← 额外的量化 kernel
#
# 折叠后：
#   silu_and_mul(x, scale, out_dtype="float8_e4m3fn") → fp8_linear(hidden_fp8, ...)
#   量化在 silu_and_mul 内部完成，节省一个 kernel
```

---

## flashinfer vs trtllm backend

| | flashinfer | trtllm |
|---|---|---|
| 默认 | ✅ | |
| 性能 | 标准 | **更快**（Triton kernel） |
| FP8 折叠 | ❌ | ✅（`out_dtype` 参数） |
| 适用场景 | 通用 | FP8 量化模型 |

---

## 写新 Fusion 的 checklist

1. ✅ 定义 `XxxConfig(TransformConfig)` —— 如果有额外参数
2. ✅ `@TransformRegistry.register("fuse_xxx")` 注册
3. ✅ `_apply`: 遍历节点 → `is_op` 匹配入口节点 → 调 pattern 匹配函数
4. ✅ `inserting_before(node)` placement —— **不要破坏图的拓扑顺序**
5. ✅ `replace_all_uses_with` —— 替换所有引用
6. ✅ `eliminate_dead_code()` + `recompile()` —— 清理
7. ✅ 传播 `meta["val"]` —— 下游 transform 依赖
8. ✅ `TransformInfo(num_matches=cnt)` —— 汇报匹配数


---
**下一节：[`15 — 图节点工具`](../stage4-fusion/15-node-utils.md)**
