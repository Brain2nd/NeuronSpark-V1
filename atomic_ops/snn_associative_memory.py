"""
SNNAssociativeMemoryLayer: 基于短时突触可塑性的脉冲联想记忆

在 TOKEN 级别操作（非帧级别），使用 parallel scan 并行化。

数学定义:
  M[t] = β_M · M[t-1] + write_gate[t] · k[t] · v[t]ᵀ   // 矩阵状态递推
  output[t] = q[t]ᵀ · M[t]                                // 线性读出

M 展平为 (D_key × D_value) 维向量后，递推退化为标量 β_M 的线性扫描，
可复用 PLIF parallel scan 基础设施并行求解。

计算流:
  (TK, batch, D) → K帧平均 → (seq_len, batch, D)
    → PLIFNode → q, k, v, gate (token 级)
    → parallel scan (β_M, gate·k⊗v) → M states
    → q · M → output → repeat K → (TK, batch, D)

复杂度: O(seq_len × D_key²), 显存 O(seq_len × D_key²)
对比 attention: O(seq_len² × D), 当 D_key << seq_len 时远优于 attention

生物对应: 短时突触可塑性 (Short-Term Synaptic Plasticity)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, surrogate

from .plif_node import PLIFNode
from .rms_norm import RMSNorm
from .parallel_scan import plif_rowparam_forward


class SNNAssociativeMemoryLayer(base.MemoryModule):
    """
    SNN 联想记忆层 — spike 驱动的矩阵状态读写（parallel scan 并行化）。

    Args:
        D: 可见维度
        D_key: 键/查询维度
        D_value: 值维度
        K: 每 token 的 SNN 帧数（用于聚合/广播）
        beta_M: 记忆衰减率
        num_layers: 总层数（用于输出缩放）
        activation_mode: v1/v2 激活模式
    """

    def __init__(
        self,
        D: int,
        D_key: int = 64,
        D_value: int = 64,
        K: int = 12,
        beta_M: float = 0.999,
        num_layers: int = 1,
        activation_mode: str = 'v2',
    ):
        super().__init__()
        self.D = D
        self.D_key = D_key
        self.D_value = D_value
        self.K = K
        self.activation_mode = activation_mode
        self.M_dim = D_key * D_value  # 展平的矩阵状态维度

        # β_M: 标量衰减率
        self.register_buffer('beta_M', torch.tensor(beta_M))

        # 输入归一化
        self.norm = RMSNorm(D)

        # 投影: D → D_key (q, k) + D_value (v) + 1 (gate logit)
        self.qkv_proj = nn.Linear(D, D_key * 2 + D_value, bias=False)
        self.gate_proj = nn.Linear(D, 1, bias=True)

        # 输出投影
        self.out_proj = nn.Linear(D_value, D, bias=False)

        # 持久状态: 展平的 M 矩阵
        self.register_memory('M_state', 0.)

        self._init_weights(num_layers)

    def _init_weights(self, num_layers):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.out_proj.weight, std=std)
        # gate 偏置初始化为负值 → sigmoid ≈ 0.1 → 大部分 token 不写入
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        self.gate_proj.weight.data.mul_(0.01)

    def forward_parallel(self, h):
        """
        并行前向传播。

        Args:
            h: (TK, batch, D) — 连续值输入

        Returns:
            output: (TK, batch, D) — 联想记忆输出（用于残差连接）
        """
        TK, batch, D = h.shape
        K = self.K
        seq_len = TK // K

        # ====== 1. K 帧平均聚合 → token 级表示 ======
        h_token = h.view(seq_len, K, batch, D).mean(dim=1)  # (seq_len, batch, D)
        h_normed = self.norm(h_token)

        # ====== 2. 投影 q, k, v + gate ======
        flat = h_normed.reshape(seq_len * batch, D)
        qkv = self.qkv_proj(flat)  # (seq_len*batch, D_key*2 + D_value)
        q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
        q = q.reshape(seq_len, batch, self.D_key)
        k = k.reshape(seq_len, batch, self.D_key)
        v = v.reshape(seq_len, batch, self.D_value)

        gate_logit = self.gate_proj(flat).reshape(seq_len, batch, 1)
        gate = torch.sigmoid(gate_logit)  # (seq_len, batch, 1)

        # L2 归一化 k（稳定外积）
        k = F.normalize(k, dim=-1)

        # ====== 3. 构造 parallel scan 输入 ======
        # 外积 k ⊗ v: (seq_len, batch, D_key, D_value) → 展平 (seq_len, batch, M_dim)
        kv_outer = (k.unsqueeze(-1) * v.unsqueeze(-2))  # (seq_len, batch, D_key, D_value)
        kv_flat = kv_outer.reshape(seq_len, batch, self.M_dim)  # (seq_len, batch, M_dim)
        b_scan = gate * kv_flat  # gate 调制的写入信号

        # a_scan: 常量 β_M 扩展到 (seq_len, batch, M_dim)
        a_scan = self.beta_M.expand(seq_len, batch, self.M_dim)

        # ====== 4. Linear recurrence via cumulative scan ======
        # M[t] = β_M · M[t-1] + b[t]
        # 用 log-space cumsum 实现并行扫描（β_M 是常量，简化为指数加权累积和）
        # M[T] = Σ_{t=0}^{T} β_M^{T-t} · b[t]

        # 初始状态
        if isinstance(self.M_state, float):
            M_init = torch.zeros(batch, self.M_dim, device=h.device, dtype=h.dtype)
        else:
            M_init = self.M_state

        # 指数加权累积和（parallel）
        # weights[t, s] = β_M^{t-s} for s <= t
        # M[t] = β_M^{t+1} · M_init + Σ_{s=0}^{t} β_M^{t-s} · b[s]
        log_beta = torch.log(self.beta_M.clamp(min=1e-8))
        # 构造 decay 权重: β_M^0, β_M^1, ..., β_M^{seq_len-1}
        decay_powers = torch.arange(seq_len, device=h.device, dtype=h.dtype)
        decay = torch.exp(log_beta * decay_powers)  # (seq_len,)

        # b_scan 按 decay 加权后做 cumsum
        # b_weighted[t] = b[t] / β_M^t
        # cumsum → M_raw[T] = Σ_{t=0}^{T} b[t] / β_M^t
        # M[T] = β_M^T · (M_init + M_raw[T])
        inv_decay = 1.0 / (decay + 1e-12)  # (seq_len,)
        b_weighted = b_scan * inv_decay[:, None, None]  # (seq_len, batch, M_dim)
        M_raw_cumsum = torch.cumsum(b_weighted, dim=0)  # (seq_len, batch, M_dim)

        # 加上初始状态并乘以 decay
        M_init_expanded = M_init.unsqueeze(0)  # (1, batch, M_dim)
        M_all = (M_init_expanded + M_raw_cumsum) * decay[:, None, None]  # (seq_len, batch, M_dim)

        # 保存最终状态
        self.M_state = M_all[-1].detach()

        # ====== 5. 读出: output[t] = q[t]ᵀ · M[t] ======
        # M_all: (seq_len, batch, D_key, D_value) reshaped
        M_matrix = M_all.reshape(seq_len, batch, self.D_key, self.D_value)
        # q: (seq_len, batch, D_key) → (seq_len, batch, 1, D_key)
        # output = q @ M: (seq_len, batch, 1, D_value) → (seq_len, batch, D_value)
        output = torch.einsum('sbk,sbkv->sbv', q, M_matrix)  # (seq_len, batch, D_value)

        # ====== 6. 输出投影 + 广播回 K 帧 ======
        output = self.out_proj(output.reshape(seq_len * batch, self.D_value))
        output = output.reshape(seq_len, batch, D)

        # 广播回 TK 帧: (seq_len, batch, D) → (seq_len, K, batch, D) → (TK, batch, D)
        output = output.unsqueeze(1).expand(-1, K, -1, -1).reshape(TK, batch, D)

        return output
