# 17 — torch.compile / torch-opt backends

> 文件: `auto_deploy/compile/compiler.py` (57行),
>       `auto_deploy/compile/backends/torch_compile.py` (32行),
>       `auto_deploy/compile/backends/torch_opt.py` (~40行)
> 问题: torch.compile + CUDA Graph 后端怎么抽象？

---

## 阅读笔记

_TODO_

## 后端注册表

```
"torch-compile"   → TorchCompileCompiler   (torch.compile only)
"torch-cudagraph" → TorchCudagraphCompiler  (CUDA Graph only)
"torch-opt"       → TorchOptCompiler        (torch.compile + CUDA Graph)
```

## 核心代码

```python
# torch_compile.py
def compile(self):
    return torch.compile(self.model, dynamic=True)

# torch_opt.py
def __init__(self, *args, **kwargs):
    torch._dynamo.config.recompile_limit = max(
        len(self.cuda_graph_batch_sizes),
        torch._dynamo.config.recompile_limit
    )
```

## 关键点

- [ ] `dynamic=True` vs `False` 在推理场景的影响？
- [ ] `recompile_limit` 为什么 >= batch sizes 数量？
- [ ] torch.compile + CUDA Graph 的叠加顺序？

## 疑问

_TODO_
