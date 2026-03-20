"""
SNN 基础组件 — 替代 SpikingJelly 依赖

提供:
  - MemoryModule: 带持久状态（膜电位 V）的 nn.Module 基类
  - reset_net: 递归重置所有神经元状态
  - SurrogateHeaviside: 替代梯度的阶跃函数（前向 Θ(x)，反向 sigmoid'）
"""

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """带持久状态管理的 Module 基类。

    register_memory(name, value) 注册的状态:
    - 跨 forward 调用持续保持（如膜电位 V）
    - reset_net() 时重置为初始值
    - 不纳入 state_dict（不是 nn.Parameter/Buffer，是瞬态运行状态）
    """

    def __init__(self):
        super().__init__()
        self._memories = {}  # name → initial_value

    def register_memory(self, name: str, value):
        """注册一个持久状态。"""
        self._memories[name] = value
        setattr(self, name, value)

    def reset_memory(self):
        """重置所有注册的状态到初始值。"""
        for name, init_val in self._memories.items():
            setattr(self, name, init_val)


def reset_net(module: nn.Module):
    """递归重置所有 MemoryModule 子模块的状态。"""
    if isinstance(module, MemoryModule):
        module.reset_memory()
    for child in module.children():
        reset_net(child)


class _SurrogateHeavisideFunction(torch.autograd.Function):
    """替代梯度阶跃函数: 前向 Heaviside Θ(x)，反向 sigmoid 导数。"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        sigmoid = torch.sigmoid(alpha * x)
        grad = grad_output * sigmoid * (1.0 - sigmoid) * alpha
        return grad, None


class SurrogateSigmoid:
    """替代梯度函数: Sigmoid surrogate。

    forward: Θ(x) = 1 if x >= 0 else 0
    backward: α·σ(αx)·(1-σ(αx))

    Args:
        alpha: 替代梯度的锐度参数（越大越接近真实阶跃，但梯度窗口越窄）
    """

    def __init__(self, alpha: float = 4.0):
        self.alpha = alpha

    def __call__(self, x):
        return _SurrogateHeavisideFunction.apply(x, self.alpha)
