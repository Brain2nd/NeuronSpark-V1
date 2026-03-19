"""
SNNAssociativeMemoryLayer: 无衰减累积 + content-based 分离的脉冲联想记忆

数学原理:
  M[T] = Σ_{t≤T} gate[t] · k[t] · v[t]ᵀ          // 无衰减累积（纯 cumsum）
  output[t] = q[t]ᵀ · M[t]                          // content-based 分离查询

  展开: output[t] = Σ_{s≤t} gate[s] · (q[t]·k[s]) · v[s]

  q·k[s] 点积从叠加态 M 中"投影"出与查询相关的分量。
  这在数学上等价于无 softmax 的线性注意力。

  距离无衰减: 位置 0 和位置 2000 的信息等权可检索。
  spike 门控: gate 控制写入稀疏性，防止 M 范数无限增长。
  k 的 L2 归一化: 每次外积 ||k·vᵀ|| 有界。
  输出 RMSNorm: 消除 M 累积范数对输出幅度的影响。

  类比流式算法: M 是 "sketch"，q 是 "query"，q·M 从压缩表示中恢复特定方向的信号。

在 TOKEN 级别操作（非帧级别），通过 cumsum 并行化。

复杂度: O(seq_len × D_key × D_value), 显存 O(seq_len × D_key × D_value)
对比 attention: O(seq_len² × D), 当 D_key·D_value << seq_len×D 时优于 attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, surrogate

from .plif_node import PLIFNode
from .rms_norm import RMSNorm
from .parallel_scan import plif_rowparam_forward


def _precompute_rope_freqs(dim, max_seq_len=8192, base=10000.0):
    """预计算 RoPE 的 cos/sin 频率表。"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    """对 q 或 k 应用 RoPE 旋转位置编码。x: (..., dim)"""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SNNAssociativeMemoryLayer(base.MemoryModule):
    """
    SNN 联想记忆层 — 无衰减累积 + content-based 分离。

    Args:
        D: 可见维度
        D_key: 键/查询维度
        D_value: 值维度
        K: 每 token 的 SNN 帧数
        num_layers: 总层数（用于输出缩放）
        activation_mode: v1/v2 激活模式
    """

    def __init__(
        self,
        D: int,
        D_key: int = 64,
        D_value: int = 64,
        K: int = 12,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        num_layers: int = 1,
        activation_mode: str = 'v2',
    ):
        super().__init__()
        self.D = D
        self.D_key = D_key
        self.D_value = D_value
        self.K = K
        self.activation_mode = activation_mode

        # RoPE 预计算（D_key 必须为偶数）
        assert D_key % 2 == 0, f"D_key must be even for RoPE, got {D_key}"
        rope_cos, rope_sin = _precompute_rope_freqs(D_key, max_seq_len, rope_base)
        self.register_buffer('rope_cos', rope_cos, persistent=False)
        self.register_buffer('rope_sin', rope_sin, persistent=False)

        # 输入归一化
        self.norm = RMSNorm(D)

        # 投影: D → q(D_key) + k(D_key) + v(D_value)
        self.qkv_proj = nn.Linear(D, D_key * 2 + D_value, bias=False)

        # 写入门控: PLIFNode (SNN 神经元驱动，输出连续泄漏量)
        self.gate_neuron = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.8,  # 高阈值 → 选择性写入
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # 输出归一化 + 投影
        self.out_norm = RMSNorm(D_value)
        self.out_proj = nn.Linear(D_value, D, bias=False)

        # 持久状态: 展平的 M 矩阵 (D_key × D_value)
        self.register_memory('M_state', 0.)

        self._init_weights(num_layers)

    def _init_weights(self, num_layers):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.out_proj.weight, std=std)
        # gate_neuron 参数由 PLIFNode 自初始化

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
        qkv = self.qkv_proj(flat)
        q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
        q = q.reshape(seq_len, batch, self.D_key)
        k = k.reshape(seq_len, batch, self.D_key)
        v = v.reshape(seq_len, batch, self.D_value)

        # ====== RoPE: 旋转位置编码应用于 q, k ======
        rope_cos = self.rope_cos[:seq_len].unsqueeze(1)  # (seq_len, 1, D_key//2)
        rope_sin = self.rope_sin[:seq_len].unsqueeze(1)  # (seq_len, 1, D_key//2)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        # PLIFNode gate: 连续泄漏量输出，跨 token 累积动力学
        gate_input = h_normed  # (seq_len, batch, D)
        beta_g = self.gate_neuron.beta
        u_g = (1.0 - beta_g) * gate_input

        v_init_g = self.gate_neuron.v
        if isinstance(v_init_g, float):
            v_init_g = torch.zeros(batch, D, device=h.device, dtype=h.dtype)

        beta_row_g = beta_g.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row_g = self.gate_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        _, V_post_g = plif_rowparam_forward(
            beta_row_g, u_g, v_th_row_g, v_init_g,
            surrogate_function=self.gate_neuron.surrogate_function,
        )
        self.gate_neuron.v = V_post_g[-1].detach()
        gate_activation = (1.0 - beta_g) * V_post_g  # (seq_len, batch, D), 连续泄漏量
        gate = gate_activation.mean(dim=-1, keepdim=True)  # (seq_len, batch, 1), D 维均值做标量门控

        # L2 归一化 k（稳定外积，确保 ||k·vᵀ|| 有界）
        k = F.normalize(k, dim=-1)

        # ====== 3. 无衰减累积: M[T] = Σ_{t≤T} gate[t] · k[t] · v[t]ᵀ ======
        # 外积 + gate 调制
        kv_outer = k.unsqueeze(-1) * v.unsqueeze(-2)  # (seq_len, batch, D_key, D_value)
        kv_gated = gate.unsqueeze(-1) * kv_outer       # (seq_len, batch, D_key, D_value)

        # cumsum = 并行累积（无衰减，等价于 β_M=1 的线性递推）
        M_all = torch.cumsum(kv_gated, dim=0)  # (seq_len, batch, D_key, D_value)

        # 加上跨序列的持久状态
        if not isinstance(self.M_state, float):
            M_all = M_all + self.M_state.unsqueeze(0)

        # 保存末尾状态（推理时跨 token 传递）
        self.M_state = M_all[-1].detach()

        # ====== 4. Content-based 分离查询 ======
        # output[t] = q[t]ᵀ · M[t] = Σ_{s≤t} gate[s] · (q[t]·k[s]) · v[s]
        output = torch.einsum('sbk,sbkv->sbv', q, M_all)  # (seq_len, batch, D_value)

        # ====== 5. 输出归一化 + 投影 + 广播回 K 帧 ======
        output = self.out_norm(output)
        output = self.out_proj(output.reshape(seq_len * batch, self.D_value))
        output = output.reshape(seq_len, batch, D)

        # 广播回 TK 帧
        output = output.unsqueeze(1).expand(-1, K, -1, -1).reshape(TK, batch, D)

        return output
