"""
SNNAttentionDecoderLayer: 完整的 SNN-Attention 解码层

对齐 SNNDecoderLayer 的双子层结构:
  子层 1: Pre-LN → SNN-Attention (替换 SNNBlock) → out_proj → 残差中心化 → 残差
  子层 2: Pre-LN → PLIFNode → SNNFFN → PonderNet → out_proj → 残差中心化 → 残差

SNN-Attention 数学:
  M[T] = Σ_{t≤T} gate[t] · k[t] · v[t]ᵀ    // 无衰减 cumsum
  output[t] = q[t]ᵀ · M[t]                    // content-based 分离
  RoPE 应用于 q, k                             // 相对位置编码
  gate 由 PLIFNode 驱动                         // spike 门控写入

在 TOKEN 级别操作, cumsum 并行化。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, surrogate

from .plif_node import PLIFNode
from .rms_norm import RMSNorm
from .snn_ffn import SNNFFN
from .snn_decoder_layer import SNNDecoderLayer
from .parallel_scan import plif_rowparam_forward


def _precompute_rope_freqs(dim, max_seq_len=8192, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SNNAttentionDecoderLayer(base.MemoryModule):
    """
    完整的 SNN-Attention 解码层, 结构对齐 SNNDecoderLayer。

    子层 1: SNN-Attention (token 级 cumsum + RoPE + PLIFNode gate)
    子层 2: SNNFFN (和 SNNDecoderLayer 完全相同)

    Args:
        D: 可见维度
        N: FFN 的状态扩展因子 (子层 2 用)
        D_ff: FFN 中间层维度
        D_key: SNN-Attention 的键/查询维度
        D_value: SNN-Attention 的值维度
        v_th_min: 动态阈值下限
        ffn_v_threshold: SNNFFN 神经元阈值
        K: 每 token SNN 帧数
        num_layers: 总层数 (用于输出缩放)
        layer_idx: 当前层索引
        activation_mode: v1/v2 激活模式
    """

    def __init__(
        self,
        D: int,
        N: int,
        D_ff: int,
        D_key: int = 64,
        D_value: int = 64,
        v_th_min: float = 0.1,
        ffn_v_threshold: float = 0.15,
        K: int = 12,
        num_layers: int = 1,
        layer_idx: int = 0,
        activation_mode: str = 'v2',
    ):
        super().__init__()
        self.D = D
        self.K = K
        self.D_key = D_key
        self.D_value = D_value
        self.activation_mode = activation_mode

        # ====== 子层 1: SNN-Attention ======
        self.attn_norm = RMSNorm(D)
        self.qkv_proj = nn.Linear(D, D_key * 2 + D_value, bias=False)
        self.gate_neuron = PLIFNode(
            dim=D, init_tau=2.0, v_threshold=0.8,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )
        self.attn_out_norm = RMSNorm(D_value)
        self.attn_out_proj = nn.Linear(D_value, D, bias=False)

        # RoPE
        assert D_key % 2 == 0
        rope_cos, rope_sin = _precompute_rope_freqs(D_key)
        self.register_buffer('rope_cos', rope_cos, persistent=False)
        self.register_buffer('rope_sin', rope_sin, persistent=False)

        # 持久状态
        self.register_memory('M_state', 0.)

        # ====== 子层 2: SNNFFN (完全复用 SNNDecoderLayer 的结构) ======
        self.ffn_norm = RMSNorm(D)
        self.input_neuron2 = PLIFNode(
            dim=D, init_tau=2.0, v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )
        self.snn_ffn = SNNFFN(
            D=D, D_ff=D_ff,
            output_v_threshold=ffn_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
            activation_mode=activation_mode,
        )
        self.ffn_out_proj = nn.Linear(D, D, bias=False)

        # PonderNet halt (子层 2)
        self.ffn_halt = nn.Linear(D, 1, bias=False)

        # ====== 初始化 ======
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.normal_(self.attn_out_proj.weight, std=std)
        nn.init.normal_(self.ffn_out_proj.weight, std=std)
        nn.init.xavier_uniform_(self.ffn_halt.weight)
        self.ffn_halt.weight.data.mul_(0.01)

    def _input_neuron_parallel(self, input_neuron, x):
        """PLIFNode parallel scan, 返回激活值。复用 SNNDecoderLayer 的逻辑。"""
        TK, batch, D = x.shape
        beta = input_neuron.beta
        u = (1.0 - beta) * x

        v_init = input_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=x.device, dtype=x.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = input_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=input_neuron.surrogate_function,
        )
        input_neuron.v = V_post[-1].detach()
        if self.activation_mode == 'v2':
            return (1.0 - beta) * V_post
        return V_post

    def _gate_neuron_parallel(self, h_normed):
        """PLIFNode gate 的 parallel scan, 返回标量门控。"""
        seq_len, batch, D = h_normed.shape
        beta_g = self.gate_neuron.beta
        u_g = (1.0 - beta_g) * h_normed

        v_init_g = self.gate_neuron.v
        if isinstance(v_init_g, float):
            v_init_g = torch.zeros(batch, D, device=h_normed.device, dtype=h_normed.dtype)

        beta_row = beta_g.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = self.gate_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        _, V_post_g = plif_rowparam_forward(
            beta_row, u_g, v_th_row, v_init_g,
            surrogate_function=self.gate_neuron.surrogate_function,
        )
        self.gate_neuron.v = V_post_g[-1].detach()
        gate_activation = (1.0 - beta_g) * V_post_g  # (seq_len, batch, D)
        return gate_activation.mean(dim=-1, keepdim=True)  # (seq_len, batch, 1)

    def _adaptive_aggregate(self, frames, halt_proj):
        """复用 SNNDecoderLayer 的 PonderNet 聚合。"""
        from .snn_decoder_layer import _fused_geometric_halt
        seq_len, K, batch, D = frames.shape
        halt_logits = halt_proj(frames).squeeze(-1)
        halt_weights = _fused_geometric_halt(halt_logits)
        aggregated = (frames * halt_weights.unsqueeze(-1)).sum(dim=1)
        steps = torch.arange(1, K + 1, device=frames.device, dtype=frames.dtype)
        expected_k = (halt_weights * steps[None, :, None]).sum(dim=1)
        ponder_cost = expected_k.mean()
        return aggregated, ponder_cost, expected_k.detach()

    def forward_parallel(self, h):
        """
        并行前向传播, 返回 (h, ponder_cost) 对齐 SNNDecoderLayer 接口。
        """
        TK, batch, D = h.shape
        K = self.K
        seq_len = TK // K

        # ====== 子层 1: SNN-Attention ======
        # K 帧平均聚合 → token 级
        h_token = h.view(seq_len, K, batch, D).mean(dim=1)
        h_normed = self.attn_norm(h_token)

        # 投影 q, k, v
        flat = h_normed.reshape(seq_len * batch, D)
        qkv = self.qkv_proj(flat)
        q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
        q = q.reshape(seq_len, batch, self.D_key)
        k = k.reshape(seq_len, batch, self.D_key)
        v = v.reshape(seq_len, batch, self.D_value)

        # RoPE
        rope_cos = self.rope_cos[:seq_len].unsqueeze(1)
        rope_sin = self.rope_sin[:seq_len].unsqueeze(1)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        # PLIFNode gate
        gate = self._gate_neuron_parallel(h_normed)

        # L2 归一化 k
        k = F.normalize(k, dim=-1)

        # 无衰减累积 M
        kv_outer = k.unsqueeze(-1) * v.unsqueeze(-2)
        kv_gated = gate.unsqueeze(-1) * kv_outer
        M_all = torch.cumsum(kv_gated, dim=0)

        if not isinstance(self.M_state, float):
            M_all = M_all + self.M_state.unsqueeze(0)
        self.M_state = M_all[-1].detach()

        # 读出 + 归一化 + 投影
        attn_out = torch.einsum('sbk,sbkv->sbv', q, M_all)
        attn_out = self.attn_out_norm(attn_out)
        res_attn = self.attn_out_proj(attn_out.reshape(seq_len * batch, self.D_value))
        res_attn = res_attn.reshape(seq_len, batch, D)
        res_attn = res_attn - res_attn.mean(dim=-1, keepdim=True)  # 残差中心化

        # 广播回 TK 帧 + 残差
        h = h + res_attn.unsqueeze(1).expand(-1, K, -1, -1).reshape(TK, batch, D)

        # ====== 子层 2: SNNFFN (和 SNNDecoderLayer 完全一致) ======
        v_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h))
        cont_ffn = self.snn_ffn.forward_parallel(v_in2)

        frames_ffn = cont_ffn.view(seq_len, K, batch, D)
        combined_ffn, pc_ffn, ek_ffn = self._adaptive_aggregate(frames_ffn, self.ffn_halt)
        res_ffn = self.ffn_out_proj(combined_ffn)
        res_ffn = res_ffn - res_ffn.mean(dim=-1, keepdim=True)

        h = h + res_ffn.repeat_interleave(K, dim=0)

        # 存储 E[K] 诊断
        with torch.no_grad():
            self._ek_min = ek_ffn.min().item()
            self._ek_max = ek_ffn.max().item()

        return h, pc_ffn
