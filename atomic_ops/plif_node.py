"""
PLIFNode: D 维固定参数 PLIF 神经元（设计文档 5.5 "普通 SNN 神经元"）

与 SelectivePLIFNode 的区别：
  SelectivePLIF: β(t), α(t), V_th(t) 由输入每步动态计算（选择性记忆）
  PLIFNode:      β, V_th 为 D 维可学习参数，训练后固定（信号转换）

每个维度有独立的可学习参数：
  β_d = sigmoid(w_d): 时间常数（衰减率）
  V_th_d: 发放阈值

动力学：
  α = (1-β) × gain, gain = 1 + 0.5·tanh(w_gain) ∈ [0.5, 1.5]
  V[t] = β · V[t-1] + α · x[t]
  s[t] = Θ(V[t] - V_th)            (surrogate gradient)
  V[t] -= V_th · s[t]              (soft reset)

增益设计：
  gain 允许每个维度独立学习 ±50% 的积分强度偏移，
  同时保持 α ∝ (1-β) 耦合，使稳态增益 α/(1-β) = gain ∈ [0.5, 1.5]，
  防止无约束漂移导致 PonderNet E[K] 膨胀。
"""

import math

import torch
import torch.nn as nn
from spikingjelly.activation_based import base, surrogate


class PLIFNode(base.MemoryModule):
    """
    D 维固定参数 PLIF 神经元。

    Args:
        dim: 神经元数量（每个维度独立参数）
        init_tau: 初始时间常数 τ（β = 1 - 1/τ）
        v_threshold: 初始发放阈值
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        dim: int,
        init_tau: float = 2.0,
        v_threshold: float = 0.5,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        # D 维可学习参数（随机初始化，每个维度独立）
        # w: 控制 β=sigmoid(w)，随机产生不同时间常数
        #    init_w ± 0.5 → β ∈ ~[sigmoid(w-0.5), sigmoid(w+0.5)]
        #    tau=2.0 时 w=0, β ∈ ~[0.38, 0.62]
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.empty(dim).normal_(init_w, 0.5))
        # w_gain: 控制 gain = 1 + 0.5·tanh(w_gain) ∈ [0.5, 1.5]
        #   α = (1-β) × gain，保持 α∝(1-β) 耦合，稳态增益有界
        #   初始 w_gain=0 → tanh(0)=0 → gain=1.0 → α=1-β（与原 PLIF 完全一致）
        self.w_gain = nn.Parameter(torch.zeros(dim))
        # v_th: 发放阈值，U[0.5x, 1.5x] 均匀分布产生维度间多样性
        self.v_th = nn.Parameter(torch.empty(dim).uniform_(
            v_threshold * 0.5, v_threshold * 1.5,
        ))
        self.surrogate_function = surrogate_function
        # 膜电位状态（functional.reset_net 时重置为 0.）
        self.register_memory('v', 0.)

    @property
    def beta(self):
        """D 维衰减率 β = sigmoid(w)，值域 (0, 1)。"""
        return torch.sigmoid(self.w)

    @property
    def gain(self):
        """D 维增益乘子 gain = 1 + 0.5·tanh(w_gain)，值域 [0.5, 1.5]。"""
        return 1.0 + 0.5 * torch.tanh(self.w_gain)

    @property
    def alpha(self):
        """D 维积分强度 α = (1-β) × gain，稳态增益 = gain ∈ [0.5, 1.5]。"""
        return (1.0 - self.beta) * self.gain

    def forward(self, x):
        """
        单步前向传播。

        V[t] = β · V[t-1] + α · x[t], spike = Θ(V-V_th), soft reset。

        Args:
            x: 输入电流, shape (batch, dim)

        Returns:
            spike: 二值脉冲, shape (batch, dim), 值域 {0, 1}
        """
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x)
        beta = self.beta
        self.v = beta * self.v + self.alpha * x
        spike = self.surrogate_function(self.v - self.v_th)
        self.v = self.v - spike * self.v_th  # soft reset
        return spike
