"""
SNNDecoderLayer: 单个 SNN 解码层（mHC 多流残差 + 动态 K 帧聚合）

  HyperConnection.pre → RMSNorm → PLIF → SNNBlock → 动态K聚合 → out_proj → HyperConnection.post
  HyperConnection.pre → RMSNorm → PLIF → SNNFFN   → 动态K聚合 → out_proj → HyperConnection.post

残差连接:
  标准残差 x + f(x) 替换为 mHC 多流残差:
    x_{l+1} = H_res @ x_l + H_post ⊗ f(H_pre @ x_l)
  H_res 约束在 Birkhoff 多面体（双随机矩阵），谱范数 ≤ 1，
  代数保证梯度不爆炸，无论网络多深。

层间状态: (seq_len, batch, n, D) — n 流残差（n=4 推荐）。
层内 SNN: 展开到 (TK, batch, D) 做 K 帧时间动力学，PonderNet 聚合回 token 级。

动态 K：
  - K 是最大步数（K_max），不是固定步数。不同 token 有效步数 ∈ [1, K_max]。
  - 每个 token 的 K 帧 SNN 输出，学习自适应停止概率 p_halt
  - PonderNet 几何分布加权：λ_k = p_k · ∏_{j<k}(1-p_j)，归一化后加权聚合
  - 不同 token 有效步数不同：简单 token 早停（E[K]小），复杂 token 用满步数
  - ponder_cost 正则化：鼓励用更少步数完成简单 token 的处理

  数学推导：
    停止概率: p_k = σ(halt_proj(frame_k))         ∈ (0,1)
    生存概率: S_k = ∏_{j=1}^{k-1} (1 - p_j)       — 到第 k 步还没停
    权重:     λ_k = p_k · S_k                       — 恰好在第 k 步停止的概率
    归一化:   λ̂_k = λ_k / Σ_k λ_k                   — 确保权重和为 1
    聚合:     output = Σ_k λ̂_k · frame_k
    代价:     E[K] = Σ_k k · λ̂_k                    — 期望步数
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, surrogate

from .plif_node import PLIFNode
from .rms_norm import RMSNorm
from .snn_block import SNNBlock
from .snn_ffn import SNNFFN
from .hyper_connection import HyperConnection
from .parallel_scan import plif_rowparam_forward
from . import spike_current_activation


# ====== Fused halt weight computation ======
# sigmoid + clamp + log1p + cumsum + exp + normalize 融合为单函数。
# 注意：不使用 @torch.compile，因其会绕过 gradient checkpoint 的
# pack/unpack hooks，导致每层泄漏 ~1 GB 显存。

def _mpd_alpha(beta_mean: float, vth_mean: float, gamma_mean: float = 1.0) -> float:
    """MPD-AGL 自适应 surrogate gradient 宽度 (IJCAI 2025)。

    surrogate gradient sg(x) = α·σ(αx)·(1-σ(αx)) 的有效区间 ≈ 2.2/α。
    若膜电位分布偏离此区间，大部分神经元 sg ≈ 0 → 梯度消失。

    公式: α = C / (√(1+β²) × γ × V_th)
      - β 增大（强积分）→ 膜电位分布更宽 → α 减小，扩大有效区间
      - γ 增大（输入 scale 增大）→ 膜电位偏移更大 → α 减小
      - V_th 增大（阈值更高）→ V_pre 分布离阈值更远 → α 减小
    C 校准使初始条件 (β=0.5, γ=1.0, V_th=0.5) → α=4.0。
    """
    C = 4.0 * math.sqrt(1.25) * 0.5  # ≈ 2.236
    width = math.sqrt(1.0 + beta_mean ** 2) * gamma_mean * max(vth_mean, 0.01)
    alpha = C / max(width, 1e-6)
    return max(2.0, min(alpha, 16.0))


def _fused_geometric_halt(halt_logits):
    """融合计算 PonderNet 几何分布停止权重。

    输入: halt_logits (seq_len, K, batch) — halt_proj 的原始输出
    输出: halt_weights (seq_len, K, batch) — 归一化几何分布权重，sum=1

    数学: p_k = σ(logit_k), S_k = ∏_{j<k}(1-p_j), λ_k = p_k·S_k, λ̂_k = λ_k/Σλ
    """
    # bf16 尾数仅 7 位，sigmoid(6.3+) 舍入为 1.0 → log1p(-1.0) = -inf → NaN
    # clamp logits 到 [-6, 6]：sigmoid(6) = 0.99609375 bf16 安全
    halt_logits = halt_logits.clamp(-6.0, 6.0)
    p_halt = torch.sigmoid(halt_logits)
    log_1_minus_p = torch.log1p(-p_halt)               # (seq_len, K, batch)
    # Exclusive cumsum: log_survive[:, k, :] = Σ_{j<k} log(1-p_j)
    # 避免 torch.cat: 用 cumsum([:, :-1]) 填充 [:, 1:]
    log_survive = torch.zeros_like(log_1_minus_p)
    log_survive[:, 1:, :] = torch.cumsum(log_1_minus_p[:, :-1, :], dim=1)
    survive = torch.exp(log_survive)                    # (seq_len, K, batch)
    halt_weights = p_halt * survive                     # λ_k = p_k · S_k
    halt_weights = halt_weights / (halt_weights.sum(dim=1, keepdim=True) + 1e-8)
    return halt_weights


class SNNDecoderLayer(base.MemoryModule):
    """
    单个 SNN 解码层（mHC 多流残差 + K 帧聚合版本）。

    层间传递 n 流残差 h_streams (seq_len, batch, n, D)。
    每个子层通过 HyperConnection 聚合 n 流 → 1 维输入，
    SNN 处理后 PonderNet 聚合 K 帧，再由 HyperConnection 分配回 n 流。

    H_res 的 Birkhoff 多面体约束（谱范数 ≤ 1）代替了原 SubLN PostNorm
    的自动增益控制，从代数层面保证深层梯度不爆炸。

    Args:
        D: 可见维度
        N: 状态扩展因子
        D_ff: FFN 中间层维度
        v_th_min: SNNBlock 动态阈值下限
        ffn_v_threshold: SNNFFN gate/up 神经元阈值
        K: 每 token 的 SNN 时间步数
        num_layers: 总层数（用于残差输出缩放 + SNNFFN down_proj 缩放）
        layer_idx: 当前层索引
        ek_floor: E[K] 下界，低于此值时产生可微分惩罚（防止 PonderNet 坍缩）
        n_hc_streams: mHC 流数量（推荐 4）
        sinkhorn_iters: Sinkhorn 迭代次数
    """

    def __init__(
        self,
        D: int,
        N: int,
        D_ff: int,
        v_th_min: float,
        ffn_v_threshold: float,
        K: int = 16,
        num_layers: int = 1,
        layer_idx: int = 0,
        ek_floor: float = 0.0,
        n_hc_streams: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.D = D
        self.K = K
        self.ek_floor = ek_floor
        self.n_hc_streams = n_hc_streams

        self.snn_block = SNNBlock(
            D=D, N=N, v_th_min=v_th_min,
        )
        self.snn_ffn = SNNFFN(
            D=D, D_ff=D_ff,
            output_v_threshold=ffn_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
        )

        # Pre-LN 分支归一化: h → RMSNorm → PLIFNode
        self.block_norm = RMSNorm(D)
        self.ffn_norm = RMSNorm(D)

        # 输入神经元: RMSNorm(h) → V_post 膜电位激活（D 维可学习 β 和 V_th）
        self.input_neuron1 = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )
        self.input_neuron2 = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # 输出投影（突触）: spike (D) → 连续空间 (D)
        self.block_out_proj = nn.Linear(D, D, bias=False)
        self.ffn_out_proj = nn.Linear(D, D, bias=False)

        # ====== 动态 K: 停止投影（突触: SNN 输出 → 停止概率） ======
        # halt_proj: D → 1，每步每 token 产生一个停止 logit
        # PonderNet 几何分布加权，替代 uniform mean 聚合
        self.block_halt = nn.Linear(D, 1, bias=True)
        self.ffn_halt = nn.Linear(D, 1, bias=True)

        # 残差输出缩放初始化（GPT-2 style: σ = 0.02 / √(2·num_layers)）
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.block_out_proj.weight, std=std)
        nn.init.normal_(self.ffn_out_proj.weight, std=std)

        # halt 初始化: 小权重 + 负偏置 → p_halt ≈ 0.03 → 接近 uniform 聚合
        # σ(-3.5) ≈ 0.029, 几何分布归一化后 λ_1/λ_K ≈ 1.5, 接近均匀
        for halt in [self.block_halt, self.ffn_halt]:
            nn.init.xavier_uniform_(halt.weight)
            halt.weight.data.mul_(0.01)
            nn.init.constant_(halt.bias, -3.5)

        # ====== mHC 超连接（替代 SubLN PostNorm + gain clamp）======
        # Birkhoff 多面体约束代替 SubLN 的自动增益控制
        self.block_hc = HyperConnection(
            n=n_hc_streams, D=D, sinkhorn_iters=sinkhorn_iters,
        )
        self.ffn_hc = HyperConnection(
            n=n_hc_streams, D=D, sinkhorn_iters=sinkhorn_iters,
        )

    def _input_neuron_parallel(self, input_neuron, x):
        """
        输入 PLIF 神经元的 parallel scan 前向传播。

        完整 PLIF 动力学: V[t] = β·V[t-1] + (1-β)·x[t], spike = Θ(V-V_th), 软重置。
        输出脉冲电流 V_th × spike 作为激活值（稀疏值域 {0, V_th}，反向梯度稠密）。

        Args:
            input_neuron: PLIFNode 实例（D 维可学习 β 和 V_th）
            x: (TK, batch, D) — 连续值输入

        Returns:
            spike_current: (TK, batch, D) — 脉冲电流（V_th × spike，值为 0 或 V_th）
        """
        TK, batch, D = x.shape

        beta = input_neuron.beta  # (D,)
        u = (1.0 - beta) * x  # (D,) broadcast → (TK, batch, D)

        v_init = input_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=x.device, dtype=x.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = input_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=input_neuron.surrogate_function,
        )
        del u  # Triton PLIF ctx 不保存 u，可安全释放

        input_neuron.v = V_post[-1].detach()
        del V_post
        return spike_current_activation(spike, v_th_row.unsqueeze(0))  # 脉冲电流作为激活值

    def _adaptive_aggregate(self, frames, halt_proj):
        """
        PonderNet 式自适应 K 帧聚合（动态 K 核心）。

        每步计算停止概率 p_k，用几何分布权重加权聚合，
        使不同 token 有不同的有效步数。

        优化: _fused_geometric_halt 将 sigmoid+log1p+cumsum+exp+normalize
        融合为单 inductor kernel（参见 snn_block._fused_modulation 同一模式）。

        数学:
          p_k = σ(halt_proj(frame_k))                 — 停止概率
          S_k = ∏_{j<k} (1-p_j)                       — 生存概率
          λ_k = p_k · S_k                             — 几何分布权重
          λ̂_k = λ_k / Σ λ_k                           — 归一化
          output = Σ λ̂_k · frame_k                    — 加权聚合
          E[K] = Σ k · λ̂_k                            — 期望步数（ponder cost）

        Args:
            frames: (seq_len, K, batch, D) — SNN 子层 K 帧输出
            halt_proj: nn.Linear(D, 1)    — 停止投影（突触）

        Returns:
            aggregated: (seq_len, batch, D) — 加权聚合结果
            ponder_cost: scalar             — 期望步数均值（正则化用）
        """
        seq_len, K, batch, D = frames.shape

        # ====== 1. halt_proj matmul（cuBLAS）+ 融合几何权重（inductor） ======
        halt_logits = halt_proj(frames).squeeze(-1)    # (seq_len, K, batch)
        halt_weights = _fused_geometric_halt(halt_logits)  # (seq_len, K, batch), 归一化

        # ====== 2. 加权聚合 ======
        # (seq_len, K, batch, 1) × (seq_len, K, batch, D) → sum → (seq_len, batch, D)
        aggregated = (frames * halt_weights.unsqueeze(-1)).sum(dim=1)

        # ====== 3. Ponder cost: E[K] per token ======
        steps = torch.arange(1, K + 1, device=frames.device, dtype=frames.dtype)
        expected_k = (halt_weights * steps[None, :, None]).sum(dim=1)  # (seq_len, batch)
        ponder_cost = expected_k.mean()               # scalar

        # ====== 4. E[K] floor: 可微分下界惩罚（遏制 PonderNet 坍缩） ======
        # 当 E[K] < floor 时产生二次惩罚，梯度流回 halt 参数使其降低停止概率
        ek_floor_cost = torch.zeros(1, device=frames.device, dtype=frames.dtype).squeeze()
        if self.ek_floor > 1.0:
            ek_floor_cost = F.relu(self.ek_floor - expected_k).pow(2).mean()

        return aggregated, ponder_cost, expected_k.detach(), ek_floor_cost

    def forward(self, h_streams):
        """前向传播入口（FSDP 兼容）。

        FSDP 通过 __call__ 触发参数聚合钩子，必须由此方法作为入口，
        而非直接调用 forward_parallel（会绕过 FSDP 钩子导致参数未聚合）。
        单卡/DDP 训练及推理可直接调用 forward_parallel。

        Args:
            h_streams: (seq_len, batch, n, D) — n 流残差

        Returns:
            h_streams: (seq_len, batch, n, D) — 更新后的 n 流残差
            ponder_cost: scalar — 两个子层的平均期望步数
            ek_floor_cost: scalar — E[K] 下界惩罚（PonderNet 坍缩遏制）
        """
        return self.forward_parallel(h_streams)

    def forward_parallel(self, h_streams):
        """
        并行前向传播：mHC 多流残差 + 脉冲电流激活 + 动态 K 帧聚合。

        流程（每子层）:
          1. HyperConnection.pre: n 流 → 1 维子层输入
          2. 展开到 TK: (seq_len, batch, D) → (TK, batch, D)
          3. RMSNorm → PLIFNode(spike_current) → SNN子层 → (TK, batch, D)
          4. PonderNet K 帧聚合 → (seq_len, batch, D)
          5. out_proj 投影
          6. HyperConnection.post: 残差混合 + 分配 → n 流

        Args:
            h_streams: (seq_len, batch, n, D) — n 流残差

        Returns:
            h_streams: (seq_len, batch, n, D) — 更新后的 n 流残差
            ponder_cost: scalar — 两个子层的平均期望步数
            ek_floor_cost: scalar — E[K] 下界惩罚
        """
        seq_len, batch, n, D = h_streams.shape
        K = self.K
        TK = seq_len * K

        # ====== MPD-AGL: 自适应 surrogate gradient 宽度 ======
        # 根据每层当前参数动态调整 alpha，使 surrogate 有效区间匹配膜电位分布
        with torch.no_grad():
            # 子层 1: input_neuron1 (after block_norm) + SNNBlock hidden_neuron
            g1 = self.block_norm.weight.data.abs().mean().item()
            b1 = self.input_neuron1.beta.mean().item()
            v1 = self.input_neuron1.v_th.abs().mean().item()
            self.input_neuron1.surrogate_function.alpha = _mpd_alpha(b1, v1, g1)

            bh = torch.sigmoid(self.snn_block.b_beta.data).mean().item()
            vh = (self.snn_block.b_th.data.abs() + self.snn_block.v_th_min).mean().item()
            self.snn_block.hidden_neuron.surrogate_function.alpha = _mpd_alpha(bh, vh)

            # 子层 2: input_neuron2 (after ffn_norm) + SNNFFN neurons
            g2 = self.ffn_norm.weight.data.abs().mean().item()
            b2 = self.input_neuron2.beta.mean().item()
            v2 = self.input_neuron2.v_th.abs().mean().item()
            self.input_neuron2.surrogate_function.alpha = _mpd_alpha(b2, v2, g2)

            # SNNFFN: gate+up 合并 scan，使用 gate_neuron 的 surrogate
            bg = self.snn_ffn.gate_neuron.beta.mean().item()
            vg = self.snn_ffn.gate_neuron.v_th.abs().mean().item()
            bu = self.snn_ffn.up_neuron.beta.mean().item()
            vu = self.snn_ffn.up_neuron.v_th.abs().mean().item()
            self.snn_ffn.gate_neuron.surrogate_function.alpha = (
                _mpd_alpha(bg, vg) + _mpd_alpha(bu, vu)) / 2.0

            # 存储诊断值
            self._alpha_input1 = self.input_neuron1.surrogate_function.alpha
            self._alpha_hidden = self.snn_block.hidden_neuron.surrogate_function.alpha
            self._alpha_input2 = self.input_neuron2.surrogate_function.alpha
            self._alpha_ffn = self.snn_ffn.gate_neuron.surrogate_function.alpha

        # ====== 子层 1: SNNBlock ======
        # HC pre: n 流 → 1 维子层输入
        h_pre, cache_block = self.block_hc.pre(h_streams)  # (seq_len, batch, D)
        # 展开到 TK: (seq_len, batch, D) → (TK, batch, D)
        h_tk = h_pre.unsqueeze(1).expand(-1, K, -1, -1).reshape(TK, batch, D)
        del h_pre

        # RMSNorm → PLIFNode(spike_current) → SNNBlock
        v_in = self._input_neuron_parallel(self.input_neuron1, self.block_norm(h_tk))
        del h_tk
        with torch.no_grad():
            self._fr_input1 = v_in.count_nonzero().item() / max(v_in.numel(), 1)
        cont_block = self.snn_block.forward_parallel(v_in)  # (TK, batch, D)
        del v_in

        # PonderNet K 帧聚合
        frames_block = cont_block.view(seq_len, K, batch, D)
        del cont_block
        combined_block, pc_block, ek_block, efc_block = self._adaptive_aggregate(
            frames_block, self.block_halt)
        del frames_block

        # 输出投影
        res_block = self.block_out_proj(combined_block)  # (seq_len, batch, D)
        del combined_block

        # HC post: 残差混合 + 分配 → n 流
        h_streams = self.block_hc.post(h_streams, res_block, cache_block)
        del res_block

        # ====== 子层 2: SNNFFN ======
        # HC pre: n 流 → 1 维子层输入
        h_pre2, cache_ffn = self.ffn_hc.pre(h_streams)  # (seq_len, batch, D)
        h_tk2 = h_pre2.unsqueeze(1).expand(-1, K, -1, -1).reshape(TK, batch, D)
        del h_pre2

        # RMSNorm → PLIFNode(spike_current) → SNNFFN
        v_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h_tk2))
        del h_tk2
        with torch.no_grad():
            self._fr_input2 = v_in2.count_nonzero().item() / max(v_in2.numel(), 1)
        cont_ffn = self.snn_ffn.forward_parallel(v_in2)  # (TK, batch, D)
        del v_in2

        frames_ffn = cont_ffn.view(seq_len, K, batch, D)
        del cont_ffn
        combined_ffn, pc_ffn, ek_ffn, efc_ffn = self._adaptive_aggregate(
            frames_ffn, self.ffn_halt)
        del frames_ffn

        res_ffn = self.ffn_out_proj(combined_ffn)  # (seq_len, batch, D)
        del combined_ffn

        # HC post: 残差混合 + 分配 → n 流
        h_streams = self.ffn_hc.post(h_streams, res_ffn, cache_ffn)
        del res_ffn

        ponder_cost = (pc_block + pc_ffn) / 2.0  # 两个子层平均
        ek_floor_cost = (efc_block + efc_ffn) / 2.0  # E[K] 下界惩罚

        # 存储 per-token E[K] 范围（诊断用，不影响计算图）
        with torch.no_grad():
            all_ek = torch.cat([ek_block.flatten(), ek_ffn.flatten()])
            self._ek_min = all_ek.min().item()
            self._ek_max = all_ek.max().item()

        return h_streams, ponder_cost, ek_floor_cost

    def single_step_forward(self, h_streams):
        """
        单步前向传播：n 流残差。

        注意：单步模式无法做动态 K 聚合（每步独立处理）。
        训练和推理均使用 forward_parallel（含动态 K 聚合）。
        此方法仅用于调试。

        Args:
            h_streams: (batch, n, D) — n 流残差

        Returns:
            h_streams: (batch, n, D) — 更新后的 n 流残差
            ponder_cost: scalar — 0.0（单步无 ponder cost）
            ek_floor_cost: scalar — 0.0
        """
        # 子层 1: SNNBlock
        h_pre, cache_block = self.block_hc.pre(h_streams)  # (batch, D)
        spike1 = self.input_neuron1(self.block_norm(h_pre))
        v_in = spike_current_activation(spike1, self.input_neuron1.v_th)
        cont_block = self.snn_block.single_step_forward(v_in)
        res_block = self.block_out_proj(cont_block)
        h_streams = self.block_hc.post(h_streams, res_block, cache_block)

        # 子层 2: SNNFFN
        h_pre2, cache_ffn = self.ffn_hc.pre(h_streams)  # (batch, D)
        spike2 = self.input_neuron2(self.ffn_norm(h_pre2))
        v_in2 = spike_current_activation(spike2, self.input_neuron2.v_th)
        cont_ffn = self.snn_ffn.single_step_forward(v_in2)
        res_ffn = self.ffn_out_proj(cont_ffn)
        h_streams = self.ffn_hc.post(h_streams, res_ffn, cache_ffn)

        _zero = torch.tensor(0.0, device=h_streams.device)
        return h_streams, _zero, _zero
