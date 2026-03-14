"""
BioSSM 层: 生物启发 SSM (Selective State Space Model) — v3 核心递推层

对标 Nemotron 3 Super 的 Mamba-2 层，保留 NeuronSpark 完整 PLIF 选择性:
  - 7 独立投影: W_in, W_β, W_α, W_th, W_gate, W_skip, W_out (完整选择性容量)
  - 双递推: K-frame 内 PLIF 递推 + 跨 token V 状态传递 (Triton 融合 PLIF scan)
  - PonderNet 动态 K 聚合
  - MPD-AGL 自适应 surrogate gradient
  - 纯二值脉冲输出 (0/1)，由 W_out + PonderNet 解码为稠密浮点
  - SelectivePLIFNode (动态 β/α/V_th)

残差形式: h_{l+1} = h_l + out_proj(PonderNet_aggregate(SNNBlock(RMSNorm(h_l))))
不再使用 HyperConnection/SubLN/lateral，标准 Pre-LN 残差。

信号流:
  h (seq_len, batch, D)
    → expand K frames → (TK, batch, D)
    → RMSNorm → PLIFNode → spike (binary 0/1)
    → SNNBlock (7 独立投影, SelectivePLIF → spike 0/1 → W_out 解码)
    → reshape (seq_len, K, batch, D)
    → PonderNet 自适应聚合 → (seq_len, batch, D)
    → out_proj(D→D) → 残差: h + out
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, surrogate

from .selective_plif import SelectivePLIFNode
from .plif_node import PLIFNode
from .rms_norm import RMSNorm
from .parallel_scan import plif_parallel_forward, plif_rowparam_forward


# ====== MPD-AGL 自适应 surrogate gradient 宽度 ======

def _mpd_alpha(beta_mean: float, vth_mean: float, gamma_mean: float = 1.0) -> float:
    """MPD-AGL 自适应 surrogate gradient 宽度。

    α = C / (√(1+β²) × γ × V_th)
    C 校准使初始条件 (β=0.5, γ=1.0, V_th=0.5) → α=4.0。
    """
    C = 4.0 * math.sqrt(1.25) * 0.5  # ≈ 2.236
    width = math.sqrt(1.0 + beta_mean ** 2) * gamma_mean * max(vth_mean, 0.01)
    alpha = C / max(width, 1e-6)
    return max(2.0, min(alpha, 16.0))


# ====== PonderNet 融合几何分布停止权重 ======

def _fused_geometric_halt(halt_logits):
    """融合计算 PonderNet 几何分布停止权重。

    输入: halt_logits (seq_len, K, batch) — halt_proj 原始输出
    输出: halt_weights (seq_len, K, batch) — 归一化几何分布权重，sum=1
    """
    halt_logits = halt_logits.clamp(-6.0, 6.0)
    p_halt = torch.sigmoid(halt_logits)
    log_1_minus_p = torch.log1p(-p_halt)
    log_survive = torch.zeros_like(log_1_minus_p)
    log_survive[:, 1:, :] = torch.cumsum(log_1_minus_p[:, :-1, :], dim=1)
    survive = torch.exp(log_survive)
    halt_weights = p_halt * survive
    halt_weights = halt_weights / (halt_weights.sum(dim=1, keepdim=True) + 1e-8)
    return halt_weights


class SNNBlock(base.MemoryModule):
    """SNN 隐状态空间 Block — 7 独立投影，完整选择性 (从 v2 移植并简化接口)。

    Args:
        D: 可见维度
        N: 状态扩展因子
        v_th_min: 动态阈值下限
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        D: int,
        N: int = 8,
        v_th_min: float = 0.1,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.D = D
        self.N = N
        self.v_th_min = v_th_min
        DN = D * N

        # 7 独立投影 (对齐 Mamba 选择性设计)
        self.W_in = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_beta_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_alpha_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_th_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_gate = layer.Linear(D, D, bias=False, step_mode='s')
        self.W_skip = layer.Linear(D, D, bias=False, step_mode='s')
        self.W_out = layer.Linear(DN, D, bias=False, step_mode='s')

        # 调制偏置
        self.b_beta = nn.Parameter(torch.empty(DN))
        self.b_alpha = nn.Parameter(torch.empty(DN))
        self.b_th = nn.Parameter(torch.empty(DN))

        # 隐状态空间神经元 (D*N 个，动态参数)
        self.hidden_neuron = SelectivePLIFNode(
            surrogate_function=surrogate_function,
            detach_reset=False,
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        """功能引导初始化。"""
        D, N = self.D, self.N
        K_ref = 16

        beta_values = torch.linspace(0.80, 0.99, N)

        # β 偏置
        b_beta_per_n = torch.log(beta_values / (1.0 - beta_values))
        self.b_beta.data.copy_(b_beta_per_n.repeat(D))
        self.b_beta.data.add_(torch.empty_like(self.b_beta).normal_(0, 0.1))

        # α 偏置
        self.b_alpha.data.normal_(0.5413, 0.1)

        # 权重初始化
        for lin in [self.W_in, self.W_gate, self.W_skip, self.W_out]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        for lin in [self.W_beta_x, self.W_alpha_x, self.W_th_x]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            lin.weight.data.mul_(0.1)

        # W_in 时间尺度缩放
        scale_per_n = torch.sqrt(1.0 - beta_values ** 2)
        scale_DN = scale_per_n.repeat(D)
        with torch.no_grad():
            self.W_in.weight.mul_(scale_DN.unsqueeze(1))

        # b_th 校准
        p_assumed = 0.15
        sigma_I_base = math.sqrt(p_assumed / 3.0)
        sigma_V_per_n = sigma_I_base * torch.sqrt(1.0 - beta_values ** (2 * K_ref))
        target_p_fire = torch.linspace(0.25, 0.08, N)
        z_scores = math.sqrt(2.0) * torch.erfinv(2.0 * (1.0 - target_p_fire) - 1.0)
        target_V_th = sigma_V_per_n * z_scores
        b_th_per_n = torch.clamp(target_V_th - self.v_th_min, min=0.05)
        self.b_th.data.copy_(b_th_per_n.repeat(D))
        self.b_th.data.add_(torch.empty_like(self.b_th).normal_(0, 0.02))

        # W_out 发放率均衡缩放
        out_scale_per_n = 1.0 / torch.sqrt(target_p_fire)
        out_scale_per_n = out_scale_per_n / out_scale_per_n.mean()
        out_scale_DN = out_scale_per_n.repeat(D)
        with torch.no_grad():
            self.W_out.weight.mul_(out_scale_DN.unsqueeze(0))

    def forward_parallel(self, spike_in_seq: torch.Tensor) -> torch.Tensor:
        """并行前向: (TK, batch, D) → (TK, batch, D)。"""
        TK, batch, D = spike_in_seq.shape
        DN = self.D * self.N

        flat = spike_in_seq.reshape(TK * batch, D)
        gate_all = torch.sigmoid(
            F.linear(flat, self.W_gate.weight).reshape(TK, batch, D)
        )
        I_skip_all = F.linear(flat, self.W_skip.weight).reshape(TK, batch, D)

        beta_all = F.linear(flat, self.W_beta_x.weight).reshape(TK, batch, DN)
        beta_all.add_(self.b_beta).sigmoid_()

        v_th_all = F.linear(flat, self.W_th_x.weight).reshape(TK, batch, DN)
        v_th_all.add_(self.b_th).abs_().add_(self.v_th_min)

        raw_alpha = F.linear(flat, self.W_alpha_x.weight).reshape(TK, batch, DN)
        raw_alpha.add_(self.b_alpha)
        u_hidden = F.linear(flat, self.W_in.weight).reshape(TK, batch, DN)
        del flat
        alpha = F.softplus(raw_alpha)
        del raw_alpha
        u_hidden.mul_(alpha)
        del alpha

        v_init_hidden = self.hidden_neuron.v
        if isinstance(v_init_hidden, float):
            v_init_hidden = torch.zeros(batch, DN, device=spike_in_seq.device,
                                        dtype=spike_in_seq.dtype)

        s_hidden, V_post_hidden, _ = plif_parallel_forward(
            beta_all, u_hidden, v_th_all, v_init_hidden, max_iter=3,
            surrogate_function=self.hidden_neuron.surrogate_function,
        )
        del beta_all, u_hidden

        self.hidden_neuron.v = V_post_hidden[-1].detach()
        del V_post_hidden

        with torch.no_grad():
            self._firing_rate_hidden = s_hidden.sum().item() / max(s_hidden.numel(), 1)

        del v_th_all
        # 纯二值脉冲 → W_out 线性解码为稠密浮点
        s_flat = s_hidden.reshape(TK * batch, DN)
        I_out_all = F.linear(s_flat, self.W_out.weight).reshape(TK, batch, D)
        del s_hidden, s_flat
        I_total_all = I_out_all * gate_all + I_skip_all

        return I_total_all


class BioSSMLayer(base.MemoryModule):
    """BioSSM 层: Pre-RMSNorm → PLIF input neuron → SNNBlock → PonderNet → out_proj → residual。

    Args:
        D: 隐藏维度
        N: 状态扩展因子
        K: 最大 SNN 时间步
        v_th_min: 动态阈值下限
        num_layers: 总层数 (用于 out_proj 缩放初始化)
        layer_idx: 当前层索引
        ek_floor: E[K] 下界惩罚阈值
    """

    def __init__(
        self,
        D: int,
        N: int = 8,
        K: int = 16,
        v_th_min: float = 0.1,
        num_layers: int = 40,
        layer_idx: int = 0,
        ek_floor: float = 0.0,
    ):
        super().__init__()
        self.D = D
        self.K = K
        self.ek_floor = ek_floor
        self.layer_idx = layer_idx

        # Pre-LN RMSNorm
        self.norm = RMSNorm(D)

        # 输入 PLIF 神经元
        self.input_neuron = PLIFNode(
            dim=D, init_tau=2.0, v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # SNNBlock 核心 (7 独立投影)
        self.snn_block = SNNBlock(D=D, N=N, v_th_min=v_th_min)

        # 输出投影 (GPT-2 style 缩放)
        self.out_proj = nn.Linear(D, D, bias=False)
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.out_proj.weight, std=std)

        # PonderNet halt 投影
        self.halt_proj = nn.Linear(D, 1, bias=True)
        nn.init.xavier_uniform_(self.halt_proj.weight)
        self.halt_proj.weight.data.mul_(0.01)
        nn.init.constant_(self.halt_proj.bias, 0.0)  # p_halt=0.5 → 二进制加权 [1/2, 1/4, 1/8, ...]

    def _input_neuron_parallel(self, x):
        """输入 PLIF 神经元 parallel scan: x → binary spike (0/1)。"""
        TK, batch, D = x.shape
        neuron = self.input_neuron

        beta = neuron.beta
        u = (1.0 - beta) * x

        v_init = neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=x.device, dtype=x.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=neuron.surrogate_function,
        )
        del u

        neuron.v = V_post[-1].detach()
        del V_post
        return spike  # 纯二值脉冲，surrogate gradient 已由 plif_rowparam_forward 内置

    def _adaptive_aggregate(self, frames):
        """PonderNet 自适应 K 帧聚合。

        Args:
            frames: (seq_len, K, batch, D)

        Returns:
            aggregated: (seq_len, batch, D)
            ponder_cost: scalar
            ek_floor_cost: scalar
        """
        seq_len, K, batch, D = frames.shape

        halt_logits = self.halt_proj(frames).squeeze(-1)  # (seq_len, K, batch)
        halt_weights = _fused_geometric_halt(halt_logits)

        aggregated = (frames * halt_weights.unsqueeze(-1)).sum(dim=1)

        steps = torch.arange(1, K + 1, device=frames.device, dtype=frames.dtype)
        expected_k = (halt_weights * steps[None, :, None]).sum(dim=1)
        ponder_cost = expected_k.mean()

        ek_floor_cost = torch.zeros(1, device=frames.device, dtype=frames.dtype).squeeze()
        if self.ek_floor > 1.0:
            ek_floor_cost = F.relu(self.ek_floor - expected_k).pow(2).mean()

        return aggregated, ponder_cost, ek_floor_cost

    def forward(self, h):
        """前向传播。

        Args:
            h: (seq_len, batch, D)

        Returns:
            h_out: (seq_len, batch, D) — 残差更新后
            ponder_cost: scalar
            ek_floor_cost: scalar
        """
        seq_len, batch, D = h.shape
        K = self.K
        TK = seq_len * K

        # MPD-AGL 自适应 surrogate gradient
        with torch.no_grad():
            g = self.norm.weight.data.abs().mean().item()
            b_in = self.input_neuron.beta.mean().item()
            v_in = self.input_neuron.v_th.abs().mean().item()
            self.input_neuron.surrogate_function.alpha = _mpd_alpha(b_in, v_in, g)

            bh = torch.sigmoid(self.snn_block.b_beta.data).mean().item()
            vh = (self.snn_block.b_th.data.abs() + self.snn_block.v_th_min).mean().item()
            self.snn_block.hidden_neuron.surrogate_function.alpha = _mpd_alpha(bh, vh)

        # 展开到 TK: (seq_len, batch, D) → (TK, batch, D)
        h_tk = h.unsqueeze(1).expand(-1, K, -1, -1).reshape(TK, batch, D)

        # Pre-LN → PLIF input neuron → binary spike (0/1)
        sc = self._input_neuron_parallel(self.norm(h_tk))
        del h_tk

        # SNNBlock 核心
        cont = self.snn_block.forward_parallel(sc)  # (TK, batch, D)
        del sc

        # PonderNet 聚合
        frames = cont.view(seq_len, K, batch, D)
        del cont
        aggregated, ponder_cost, ek_floor_cost = self._adaptive_aggregate(frames)
        del frames

        # 输出投影 + 残差
        out = self.out_proj(aggregated)
        h_out = h + out

        return h_out, ponder_cost, ek_floor_cost
