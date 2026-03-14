"""
BioAttention 层: GQA 注意力锚点层

对标 Nemotron 3 Super 的 Attention anchor 设计:
  - Grouped-Query Attention: 8 Q-heads, 2 KV-heads, head_dim=128
  - 无位置编码 (论文: "we omit positional embeddings")
    → BioSSM 递推已在 hidden state 中编码位置信息，Attention 层不需要 RoPE
  - FlashAttention via scaled_dot_product_attention
  - KV cache 用于推理时自回归生成
  - 无 K-frame 展开，无 PonderNet，直接操作 token 级残差流

信号流:
  h (seq_len, batch, D)
    → RMSNorm → Q/K/V proj → GQA → causal attention → O proj
    → 残差: h + attn_out
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm


class BioAttentionLayer(nn.Module):
    """BioAttention 锚点层: Pre-RMSNorm → GQA → causal attn → O proj → residual。

    无位置编码，对齐 Nemotron 3 Super: "Consistent with prior Nemotron models,
    we omit positional embeddings, dropout, and bias terms in linear layers."

    Args:
        D: 隐藏维度
        n_q_heads: Q 头数
        n_kv_heads: KV 头数 (GQA 分组)
        head_dim: 注意力头维度
        max_seq_len: 最大序列长度 (KV cache 用)
        num_layers: 总层数 (用于 out_proj 缩放初始化)
        layer_idx: 当前层索引
    """

    def __init__(
        self,
        D: int,
        n_q_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 128,
        max_seq_len: int = 2048,
        num_layers: int = 40,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.D = D
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_groups = n_q_heads // n_kv_heads
        self.layer_idx = layer_idx

        # Pre-LN RMSNorm
        self.norm = RMSNorm(D)

        # Q/K/V 投影 (无 bias，对齐论文)
        self.q_proj = nn.Linear(D, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(D, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(D, n_kv_heads * head_dim, bias=False)

        # 输出投影 (GPT-2 style 缩放，无 bias)
        self.o_proj = nn.Linear(n_q_heads * head_dim, D, bias=False)
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.o_proj.weight, std=std)

        # KV cache (推理用)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))

    def reset_cache(self):
        """重置 KV cache。"""
        self.k_cache = None
        self.v_cache = None

    def forward(self, h, use_cache: bool = False):
        """前向传播。

        Args:
            h: (seq_len, batch, D) — 注意 seq_len 在前
            use_cache: 是否使用/更新 KV cache

        Returns:
            h_out: (seq_len, batch, D)
        """
        seq_len, batch, D = h.shape

        # Pre-LN
        h_normed = self.norm(h)

        # 转为 (batch, seq_len, D) 方便注意力计算
        x = h_normed.permute(1, 0, 2)  # (batch, seq_len, D)

        # Q/K/V 投影
        q = self.q_proj(x).view(batch, seq_len, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # KV cache
        if use_cache:
            if self.k_cache is not None:
                k = torch.cat([self.k_cache, k], dim=2)
                v = torch.cat([self.v_cache, v], dim=2)
            self.k_cache = k.detach()
            self.v_cache = v.detach()

        # GQA: expand KV heads to match Q heads
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention (FlashAttention backend)
        # 无位置编码 — BioSSM 递推已在残差流中编码位置信息
        is_causal = not use_cache or self.k_cache is None or self.k_cache.shape[2] == k.shape[2]
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=0.0,
        )

        # 合并头
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # O 投影
        out = self.o_proj(attn_out)

        # 转回 (seq_len, batch, D) + 残差
        out = out.permute(1, 0, 2)
        h_out = h + out

        return h_out
