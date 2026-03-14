"""
BioLatentMoE 层: 潜空间路由 Mixture-of-Experts (Latent MoE)

对标 Nemotron 3 Super 的 LatentMoE 设计:
  - 全维度路由: router 在全维度 d 操作 (gating network stays in full d)
  - 潜空间专家: token 在低维 (latent_dim=256) 做专家计算
  - Sigmoid router + auxiliary-loss-free 负载均衡 (EMA 偏置调整)
  - 标准 load balance loss (coefficient=1e-4) 作为补充
  - 共享专家: 所有 token 共享一个大专家 (shared_expert_hidden=2048)，在全维度 D 操作
  - spike_current 激活: 无记忆单步 PLIF 替代传统激活函数

信号流:
  h (seq_len, batch, D)
    → RMSNorm
    ├─ router(D→num_experts) → sigmoid → top-k 选择   [全维度路由]
    ├─ latent_down(D→latent_dim) → 选中专家计算 → latent_up(latent→D)  [潜空间专家]
    ├─ shared_expert: FC1(D→shared_hidden) → spike_current → FC2(shared_hidden→D)
    → combine → 残差: h + out
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm
from . import spike_current_activation


class _SpikeCurrent(nn.Module):
    """无记忆 spike_current 激活 (β=0 单步 PLIF)。

    spike = Θ(x - v_th), output = v_th × spike
    前向稀疏，反向梯度通过 spike_current_activation 稠密传播。
    """

    def __init__(self, dim: int, v_threshold: float = 0.3):
        super().__init__()
        self.v_th = nn.Parameter(torch.empty(dim).uniform_(
            v_threshold * 0.5, v_threshold * 1.5,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Heaviside spike (硬阈值，反向用 spike_current_activation 的 surrogate)
        spike = (x >= self.v_th).to(x.dtype)
        # v_th 需要广播到 spike 形状: (dim,) → (1, ..., dim)
        v_th = self.v_th
        for _ in range(spike.dim() - v_th.dim()):
            v_th = v_th.unsqueeze(0)
        return spike_current_activation(spike, v_th)


class Expert(nn.Module):
    """单个潜空间专家: 门控 MLP (照抄 NVIDIA gated_linear_unit 结构)。

    fc1(latent→2×hidden) → chunk → spike_current(gate) × value → fc2(hidden→latent)
    结构照抄 NVIDIA，SiLU 替换为 spike_current。
    """

    def __init__(self, latent_dim: int, expert_hidden: int):
        super().__init__()
        # 单个 fc1 输出 2×hidden，chunk 后分为 gate 和 value (照抄 NVIDIA)
        self.fc1 = nn.Linear(latent_dim, 2 * expert_hidden, bias=False)
        self.activation = _SpikeCurrent(expert_hidden, v_threshold=0.3)
        self.fc2 = nn.Linear(expert_hidden, latent_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)                          # (N, 2 * expert_hidden)
        x_gate, x_linear = x.chunk(2, dim=-1)    # 各 (N, expert_hidden)
        return self.fc2(self.activation(x_gate) * x_linear)


class SharedExpert(nn.Module):
    """共享专家: 门控 MLP (照抄 NVIDIA gated_linear_unit 结构)。

    fc1(D→2×shared_hidden) → chunk → spike_current(gate) × value → fc2(shared_hidden→D)
    在全维度 (D) 操作，所有 token 都经过此专家。
    """

    def __init__(self, D: int, shared_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(D, 2 * shared_hidden, bias=False)
        self.activation = _SpikeCurrent(shared_hidden, v_threshold=0.3)
        self.fc2 = nn.Linear(shared_hidden, D, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)                          # (N, 2 * shared_hidden)
        x_gate, x_linear = x.chunk(2, dim=-1)    # 各 (N, shared_hidden)
        return self.fc2(self.activation(x_gate) * x_linear)


class BioLatentMoELayer(nn.Module):
    """BioLatentMoE 层: 全维度路由 + 潜空间专家 + 共享专家 + 残差。

    对齐论文: "all non-routed computations—including the routing gate (gating
    network), shared expert computation, and non-expert layers—remain in the
    full hidden dimension d"

    Args:
        D: 隐藏维度
        latent_dim: 潜空间维度 (专家计算用)
        num_experts: 专家数量
        top_k: 每 token 激活的专家数
        expert_hidden: 专家隐层维度
        shared_expert_hidden: 共享专家隐层维度
        num_layers: 总层数 (用于 out_proj 缩放初始化)
        layer_idx: 当前层索引
        aux_loss_coeff: 辅助 load balance loss 系数
        ema_update_rate: EMA 偏置调整更新率
    """

    def __init__(
        self,
        D: int,
        latent_dim: int = 256,
        num_experts: int = 32,
        top_k: int = 4,
        expert_hidden: int = 1024,
        shared_expert_hidden: int = 2048,
        num_layers: int = 40,
        layer_idx: int = 0,
        aux_loss_coeff: float = 1e-4,
        ema_update_rate: float = 1e-3,
    ):
        super().__init__()
        self.D = D
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.ema_update_rate = ema_update_rate
        self.layer_idx = layer_idx

        # Pre-LN RMSNorm
        self.norm = RMSNorm(D)

        # 潜空间投影
        self.latent_down = nn.Linear(D, latent_dim, bias=False)
        self.latent_up = nn.Linear(latent_dim, D, bias=False)

        # Sigmoid router: 在全维度 D 操作 (论文: gating network stays in full d)
        self.router = nn.Linear(D, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)  # 零初始化: 初始均匀路由

        # 路由器 EMA 偏置 (auxiliary-loss-free 负载均衡)
        self.register_buffer('router_bias', torch.zeros(num_experts))
        self.register_buffer('expert_load_ema', torch.ones(num_experts) / num_experts)

        # 专家
        self.experts = nn.ModuleList([
            Expert(latent_dim, expert_hidden) for _ in range(num_experts)
        ])

        # 共享专家
        self.shared_expert = SharedExpert(D, shared_expert_hidden)

        # 共享专家 sigmoid gate (对齐 NVIDIA: gate_weight 动态调节共享专家贡献)
        self.shared_gate = nn.Linear(D, 1, bias=False)
        nn.init.zeros_(self.shared_gate.weight)  # 初始 sigmoid(0)=0.5，共享专家贡献 50%

        # 输出投影 (GPT-2 style 缩放)
        self.out_proj = nn.Linear(D, D, bias=False)
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.out_proj.weight, std=std)

        # 潜空间投影初始化
        nn.init.kaiming_uniform_(self.latent_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.latent_up.weight, a=math.sqrt(5))

    def _route(self, h_flat: torch.Tensor):
        """Sigmoid router (全维度 D) + top-k 选择 + EMA 偏置调整。

        Args:
            h_flat: (N_tokens, D) — 全维度输入

        Returns:
            expert_indices: (N_tokens, top_k) — 选中的专家索引
            expert_weights: (N_tokens, top_k) — 归一化权重
            load_balance_loss: scalar
        """
        # Sigmoid router logits + EMA 偏置 (auxiliary-loss-free 负载均衡)
        logits = self.router(h_flat)  # (N_tokens, num_experts)
        scores = torch.sigmoid(logits + self.router_bias)

        # Top-k 选择
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)

        # 归一化权重 (在 top-k 内归一化)
        weights = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # 标准 load balance loss
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability for expert i
        # LB loss = num_experts * Σ f_i * P_i
        with torch.no_grad():
            # 统计每个专家被选中的频率
            expert_mask = torch.zeros(
                h_flat.shape[0], self.num_experts,
                device=h_flat.device, dtype=h_flat.dtype,
            )
            expert_mask.scatter_(1, top_k_indices, 1.0)
            f = expert_mask.mean(dim=0)  # (num_experts,)

        P = scores.mean(dim=0)  # (num_experts,) — 需要梯度
        lb_loss = self.num_experts * (f * P).sum() * self.aux_loss_coeff

        # EMA 偏置更新 (训练模式，不需要梯度)
        if self.training:
            with torch.no_grad():
                self.expert_load_ema.mul_(1.0 - self.ema_update_rate).add_(
                    f * self.ema_update_rate
                )
                # 偏置: 低负载专家正偏置 (鼓励选择)，高负载专家负偏置 (抑制选择)
                mean_load = self.expert_load_ema.mean()
                self.router_bias.copy_(
                    (mean_load - self.expert_load_ema) * 10.0  # 放大信号
                )

        return top_k_indices, weights, lb_loss

    def forward(self, h):
        """前向传播。

        Args:
            h: (seq_len, batch, D)

        Returns:
            h_out: (seq_len, batch, D)
            load_balance_loss: scalar
        """
        seq_len, batch, D = h.shape
        N_tokens = seq_len * batch

        # Pre-LN
        h_normed = self.norm(h)

        # 全维度路由 (论文: gating network in full d)
        h_flat = h_normed.reshape(N_tokens, D)
        expert_indices, expert_weights, lb_loss = self._route(h_flat)

        # 潜空间投影 (仅用于专家计算)
        latent = self.latent_down(h_flat)  # (N_tokens, latent_dim)

        # 专家计算 (逐专家处理，在潜空间)
        expert_output = torch.zeros_like(latent)  # (N_tokens, latent_dim)

        for i in range(self.num_experts):
            # 找到选中此专家的 token
            mask = (expert_indices == i).any(dim=-1)  # (N_tokens,)
            if not mask.any():
                continue

            # 获取对应 token 和权重
            token_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if token_indices.dim() == 0:
                token_indices = token_indices.unsqueeze(0)

            expert_input = latent[token_indices]  # (n, latent_dim)
            expert_out = self.experts[i](expert_input)  # (n, latent_dim)

            # 获取此专家在 top-k 中的权重
            # expert_indices[token_indices]: (n, top_k)
            expert_idx_for_tokens = expert_indices[token_indices]
            weight_mask = (expert_idx_for_tokens == i).to(expert_out.dtype)
            expert_w = (expert_weights[token_indices] * weight_mask).sum(dim=-1, keepdim=True)

            expert_output[token_indices] += expert_out * expert_w

        # 潜空间 → 全维度
        routed_out = self.latent_up(expert_output)  # (N_tokens, D)
        routed_out = routed_out.reshape(seq_len, batch, D)

        # 共享专家 (全维度 D) + sigmoid gate
        shared_out = self.shared_expert(h_flat)  # (N_tokens, D)
        gate = torch.sigmoid(self.shared_gate(h_flat))  # (N_tokens, 1)
        shared_out = (shared_out * gate).reshape(seq_len, batch, D)

        # 组合 + 输出投影 + 残差
        combined = routed_out + shared_out
        out = self.out_proj(combined)
        h_out = h + out

        return h_out, lb_loss
