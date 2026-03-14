"""
MTP Head: Multi-Token Prediction 共享权重预测头

照抄 NVIDIA Nemotron 3 Super 的 MTP 设计:
  - 分离式 enorm + hnorm 归一化 → concat → Linear(2D→D)
  - 门控 MLP trunk (照抄 NVIDIA gated_linear_unit 结构)
  - 单一共享 combine+trunk 模块，被多个 offset 复用
  - final_layernorm → shared LM head → CE loss

NVIDIA 原始 MTP 内部模型是完整 TransformerLayer (含 self-attention + MLP)。
当前实现用门控 MLP 简化 (无 self-attention)。

信号流:
  h (batch, seq_len, D) + target_ids (batch, seq_len) + lm_head_weight + embed_weight
    → For each offset:
        enorm(emb) + hnorm(h) → cat → proj(2D→D)
        → 门控 MLP: fc1(D→2×mlp_hidden) → chunk → relu²(gate) × value → fc2 → D
        → output_norm → shared lm_head → CE loss
    → mtp_loss = mean(losses)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm


class MTPHead(nn.Module):
    """Multi-Token Prediction 共享权重预测头。

    照抄 NVIDIA MTP 设计: enorm/hnorm 分离归一化 + 门控 MLP + 共享 LM head。

    Args:
        D: 隐藏维度
        vocab_size: 词表大小
        n_heads: 预测头数 (NVIDIA Super 120B 默认 1)
        mlp_hidden: 门控 MLP 中间维度 (默认 2×D)
    """

    def __init__(self, D: int, vocab_size: int, n_heads: int = 1,
                 mlp_hidden: int = None):
        super().__init__()
        self.D = D
        self.vocab_size = vocab_size
        self.n_heads = n_heads

        # 分离式归一化 (照抄 NVIDIA: enorm + hnorm 各自归一化后再 concat)
        self.enorm = RMSNorm(D)   # embedding 归一化
        self.hnorm = RMSNorm(D)   # hidden state 归一化
        self.combine_proj = nn.Linear(2 * D, D, bias=False)

        # 门控 MLP trunk (照抄 NVIDIA gated_linear_unit 结构)
        mlp_hidden = mlp_hidden or 2 * D
        self.trunk_fc1 = nn.Linear(D, 2 * mlp_hidden, bias=False)
        self.trunk_fc2 = nn.Linear(mlp_hidden, D, bias=False)

        # 输出 RMSNorm (照抄 NVIDIA final_layernorm)
        self.output_norm = RMSNorm(D)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.combine_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.trunk_fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.trunk_fc2.weight, a=math.sqrt(5))

    def forward(
        self,
        h: torch.Tensor,
        target_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
        embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """计算 MTP 辅助 loss。

        Args:
            h: (batch, seq_len, D) — 主模型隐状态
            target_ids: (batch, seq_len) — 目标 token IDs
            lm_head_weight: (vocab_size, D) — 独立 LM head 权重 (un-tied)
            embed_weight: (vocab_size, D) — embedding 权重

        Returns:
            mtp_loss: scalar — 多 offset CE loss 均值
        """
        batch, seq_len, D = h.shape
        losses = []

        for head_idx in range(self.n_heads):
            offset = head_idx + 1

            if seq_len <= offset + 1:
                continue

            # h_trunc: 去掉末尾 offset 个位置
            h_trunc = h[:, :-(offset + 1)]  # (batch, S, D)

            # target embedding
            target_tokens = target_ids[:, offset:-1]  # (batch, S)
            target_emb = F.embedding(target_tokens, embed_weight)  # (batch, S, D)

            # 分离式归一化 + concat + proj (照抄 NVIDIA: enorm/hnorm)
            h_normed = self.hnorm(h_trunc)
            emb_normed = self.enorm(target_emb)
            combined = torch.cat([h_normed, emb_normed], dim=-1)  # (batch, S, 2D)
            combined = self.combine_proj(combined)  # (batch, S, D)

            # 门控 MLP trunk (照抄 NVIDIA gated_linear_unit)
            x = self.trunk_fc1(combined)                    # (batch, S, 2×mlp_hidden)
            x_gate, x_linear = x.chunk(2, dim=-1)          # 各 (batch, S, mlp_hidden)
            trunk_out = self.trunk_fc2(
                F.relu(x_gate).square() * x_linear          # relu² gate
            )                                                # (batch, S, D)

            # Output norm + shared LM head (照抄 NVIDIA final_layernorm)
            trunk_out = self.output_norm(trunk_out)
            logits = F.linear(trunk_out, lm_head_weight)  # (batch, S, vocab_size)

            # CE loss
            mtp_targets = target_ids[:, offset + 1:]  # (batch, S)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                mtp_targets.reshape(-1),
                ignore_index=0,
            )
            losses.append(loss)

        if not losses:
            return torch.zeros(1, device=h.device, dtype=h.dtype).squeeze()

        return sum(losses) / len(losses)
