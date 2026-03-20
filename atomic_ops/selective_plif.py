"""
SelectivePLIFNode: 动态参数的 PLIF 神经元

与标准 PLIFNode 的区别：
- β(t), α(t), V_th(t) 作为外部参数每步传入，不是内部 nn.Parameter
- 本神经元无可训练参数，所有学习发生在 SNNBlock 的调制网络中
- 仅支持 soft reset

状态方程：
  V[t] = β(t) · V[t-1] + α(t) · I[t]
  s[t] = Θ(V[t] - V_th(t))           (surrogate gradient)
  V[t] -= V_th(t) · s[t]              (soft reset)
"""

import torch
from .snn_base import MemoryModule, SurrogateSigmoid


class SelectivePLIFNode(MemoryModule):
    """
    隐状态空间的核心神经元。

    接收外部动态计算的 β(t), α(t), V_th(t)，执行：
      charge → fire → soft reset

    Args:
        surrogate_function: surrogate gradient 函数，默认 Sigmoid(alpha=4.0)
        detach_reset: 是否在 reset 时 detach spike，默认 False
    """

    def __init__(
        self,
        surrogate_function=SurrogateSigmoid(alpha=4.0),
        detach_reset: bool = False,
    ):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.register_memory('v', 0.)

    def single_step_forward(
        self,
        x: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        v_th: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步前向传播。

        Args:
            x:    输入电流 I[t],   shape (batch, D*N)
            beta: 衰减率 β(t),    shape (batch, D*N), 值域 (0, 1)
            alpha: 写入增益 α(t),  shape (batch, D*N), 值域 R+
            v_th: 动态阈值 V_th(t), shape (batch, D*N), 值域 R+

        Returns:
            spike: 脉冲 s[t], shape (batch, D*N)
        """
        # 首步将 v 从 float 扩展为与输入同形的张量
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x)

        # Charge: V[t] = β(t) · V[t-1] + α(t) · I[t]
        self.v = beta * self.v + alpha * x

        # Fire: spike = Θ(V - V_th)
        spike = self.surrogate_function(self.v - v_th)

        # Soft Reset: V[t] -= V_th · s[t]
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = self.v - spike_d * v_th

        return spike
