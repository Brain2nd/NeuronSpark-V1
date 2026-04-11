"""
SNN 基础组件。

组件:
  MemoryModule: 有状态模块基类（状态注册/重置）
  BaseNode: 脉冲神经元基类（膜电位管理）
  Sigmoid: Surrogate gradient (nn.Module, Heaviside forward + sigmoid backward)
  Linear: nn.Linear + step_mode 属性
  reset_net: 递归重置网络状态

命名空间:
  surrogate.Sigmoid(alpha=4.0)
  functional.reset_net(net)
  layer.Linear(D, D_ff, bias=False, step_mode='s')
"""

import copy
import torch
import torch.nn as nn


# ============================================================
# MemoryModule
# ============================================================

class MemoryModule(nn.Module):
    """有状态模块基类：提供 register_memory / reset 状态管理。

    提供 register_memory / reset 状态管理:
    - _memories: {name: current_value}
    - _memories_rv: {name: reset_value (deepcopy)}
    - __getattr__ / __setattr__ / __delattr__ 代理 _memories
    - reset() 用 deepcopy(_memories_rv[name]) 还原
    """

    def __init__(self):
        super().__init__()
        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        """注册有状态变量。reset() 时会被还原为 value 的 deepcopy。"""
        self._memories[name] = value
        self._memories_rv[name] = copy.deepcopy(value)

    def reset(self):
        """重置所有有状态变量为注册时的值。"""
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value):
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            super().__delattr__(name)


# ============================================================
# BaseNode
# ============================================================

class BaseNode(MemoryModule):
    """脉冲神经元基类：膜电位状态 + v_float_to_tensor。

    保留当前仓库实际使用的全部构造参数和属性。
    """

    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function=None, detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq

    def v_float_to_tensor(self, x: torch.Tensor):
        """将标量膜电位扩展为与输入同形的张量。"""
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


# ============================================================
# Sigmoid
# ============================================================

class _SigmoidGrad(torch.autograd.Function):
    """Heaviside forward + sigmoid surrogate backward。"""

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x)  # 保留输入 dtype

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        sgax = (x * alpha).sigmoid_()
        return grad_output * (1. - sgax) * sgax * alpha, None


class Sigmoid(nn.Module):
    """Surrogate gradient: Heaviside forward, sigmoid backward.

    类名必须是 Sigmoid — parallel_scan.py 按 type().__name__ 判断 Triton 快路径。
    """

    def __init__(self, alpha=4.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return _SigmoidGrad.apply(x, self.alpha)
        else:
            return (x * self.alpha).sigmoid()


# ============================================================
# Linear
# ============================================================

class Linear(nn.Linear):
    """nn.Linear + step_mode 属性（保留接口契约）。"""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, step_mode='s'):
        super().__init__(in_features, out_features, bias)
        self.step_mode = step_mode


# ============================================================
# reset_net
# ============================================================

def reset_net(net: nn.Module):
    """重置网络中所有 MemoryModule 的状态。"""
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()


# ============================================================
# 命名空间导出 — 让下游代码零改动
# ============================================================

class surrogate:
    """命名空间：surrogate.Sigmoid(alpha=4.0) 保持原调用方式。"""
    Sigmoid = Sigmoid


class functional:
    """命名空间：functional.reset_net(net) 保持原调用方式。"""
    reset_net = staticmethod(reset_net)


class layer:
    """命名空间：layer.Linear(D, D_ff, bias=False, step_mode='s') 保持原调用方式。"""
    Linear = Linear
