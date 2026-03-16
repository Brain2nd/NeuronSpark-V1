# coding=utf-8
# Copyright 2024 HuggingFace Inc. team.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Modified by NeuronSpark: Mamba → BioSSM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch NeuronSpark model (based on NVIDIA NemotronH, Mamba replaced by BioSSM)."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache  # we need __iter__ and __len__ of pkv
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)

from atomic_ops.bio_ssm_layer import BioSSMLayer
from atomic_ops.mtp_head import MTPHead


logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


_CHECKPOINT_FOR_DOC = "NeuronSpark/NeuronSpark-V3"
_CONFIG_FOR_DOC = "NeuronSparkConfig"


# ============================================================
# NeuronSparkConfig (替代 NemotronHConfig, 内联定义)
# ============================================================

class NeuronSparkConfig(PretrainedConfig):
    model_type = "neuronspark"

    def __init__(
        self,
        vocab_size=6144,
        tie_word_embeddings=False,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=40,
        hybrid_override_pattern="SESESESE*ESESESESE*ESESESESE*ESESESESESE",
        num_attention_heads=8,
        head_dim=128,
        num_key_value_heads=2,
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        rescale_prenorm_residual=True,
        # MoE (照抄 NVIDIA)
        n_routed_experts=32,
        moe_intermediate_size=1024,
        moe_shared_expert_intermediate_size=2048,
        num_experts_per_tok=4,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        # BioSSM (替代 Mamba)
        ssm_N=8,
        ssm_K=16,
        ssm_v_th_min=0.1,
        ssm_ek_floor=4.0,
        # MTP
        n_mtp_heads=1,
        max_seq_len=512,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hybrid_override_pattern = hybrid_override_pattern
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.rescale_prenorm_residual = rescale_prenorm_residual
        # MoE
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        # BioSSM
        self.ssm_N = ssm_N
        self.ssm_K = ssm_K
        self.ssm_v_th_min = ssm_v_th_min
        self.ssm_ek_floor = ssm_ek_floor
        # MTP
        self.n_mtp_heads = n_mtp_heads
        self.max_seq_len = max_seq_len
        # HF _attn_implementation default
        if not hasattr(self, '_attn_implementation'):
            self._attn_implementation = "sdpa"

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        mapping = {'S': 'ssm', 'E': 'moe', '*': 'attention', '-': 'mlp'}
        return [mapping[c] for c in self.hybrid_override_pattern]


# ============================================================
# NeuronSparkDynamicCache (renamed from HybridMambaAttentionDynamicCache)
# ============================================================

# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py
class NeuronSparkDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the ssm cache.

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for ssm cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For ssm layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` and `ssm_states` are placeholders (empty tensors).
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        super().__init__()
        self.dtype = dtype
        self.hybrid_override_pattern = config.hybrid_override_pattern
        self.has_previous_state = False  # only used by ssm
        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if self.hybrid_override_pattern[i] == "S":
                # SSM layer — placeholder empty tensors (BioSSMLayer manages its own state)
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
            else:
                # Attention or MLP/MoE layer
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                if self.hybrid_override_pattern[i] == "*":
                    self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        if not self.transformer_layers:
            return 0
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("NeuronSparkDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("NeuronSparkDynamicCache does not have a legacy cache equivalent.")

    def reset(self):
        for i in range(len(self.conv_states)):
            if self.conv_states[i].numel() > 0:
                self.conv_states[i].zero_()
            if self.ssm_states[i].numel() > 0:
                self.ssm_states[i].zero_()


# ============================================================
# BioSSMMixer (替换 NemotronHMamba2Mixer)
# ============================================================

class BioSSMMixer(nn.Module):
    """替换 NemotronHMamba2Mixer: 包装 BioSSMLayer 为 NVIDIA mixer 接口。"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.bio_ssm = BioSSMLayer(
            D=config.hidden_size, N=config.ssm_N, K=config.ssm_K,
            v_th_min=config.ssm_v_th_min, num_layers=config.num_hidden_layers,
            layer_idx=layer_idx, ek_floor=config.ssm_ek_floor,
        )
        self.ponder_cost = None
        self.ek_floor_cost = None

    def forward(self, hidden_states, **kwargs):
        # BioSSMLayer expects (seq_len, batch, D), NVIDIA uses (batch, seq_len, D)
        h = hidden_states.permute(1, 0, 2).contiguous()
        h_out, pc, efc = self.bio_ssm(h)
        out = h_out - h  # strip BioSSMLayer internal residual
        self.ponder_cost = pc
        self.ek_floor_cost = efc
        return out.permute(1, 0, 2).contiguous()


# ============================================================
# NeuronSparkRMSNorm (renamed from NemotronHRMSNorm)
# ============================================================

class NeuronSparkRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        NeuronSparkRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Weights are in float32
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


# ============================================================
# NeuronSparkMLP (renamed from NemotronHMLP, code verbatim)
# ============================================================

# Copied from transformers.models.nemotron.modeling_nemotron Nemotron->NeuronSpark
class NeuronSparkMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


# ============================================================
# NeuronSparkTopkRouter (renamed from NemotronHTopkRouter, code verbatim)
# ============================================================

class NeuronSparkTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


# ============================================================
# NeuronSparkMOE (renamed from NemotronHMOE, code verbatim)
# ============================================================

class NeuronSparkMOE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                NeuronSparkMLP(config, intermediate_size=config.moe_intermediate_size, layer_idx=layer_idx)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NeuronSparkTopkRouter(config)
        self.shared_experts = NeuronSparkMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size, layer_idx=layer_idx
        )

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
            else:
                # Local empty expert: no-op compute that still marks params as used.
                expert_dtype = expert.down_proj.weight.dtype
                dummy_out = expert(torch.zeros_like(hidden_states[0]).unsqueeze(0).to(expert_dtype))
                final_hidden_states = final_hidden_states + dummy_out

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# ============================================================
# repeat_kv (kept as-is)
# ============================================================

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================
# NeuronSparkAttention (renamed from NemotronHAttention, code verbatim)
# ============================================================

class NeuronSparkAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_embeddings: Tuple[torch.Tensor, torch.Tensor], #TODO
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[NeuronSparkDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        #attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# ============================================================
# NeuronSparkFlashAttention2 (renamed from NemotronHFlashAttention2, code verbatim)
# ============================================================

# Adapted from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->NeuronSpark
class NeuronSparkFlashAttention2(NeuronSparkAttention):
    """
    NeuronSpark flash attention module. This module inherits from `NeuronSparkAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[NeuronSparkDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self.config, "sliding_window", None),
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        #attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ============================================================
# NeuronSparkSdpaAttention (renamed from NemotronHSdpaAttention, code verbatim)
# ============================================================

# Adapted from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->NeuronSpark
class NeuronSparkSdpaAttention(NeuronSparkAttention):
    """
    NeuronSpark attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `NeuronSparkAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from NeuronSparkAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[NeuronSparkDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "NeuronSparkModel is using NeuronSparkSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if self.is_causal and causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# ============================================================
# NEURONSPARK_ATTENTION_CLASSES (renamed from NEMOTRONH_ATTENTION_CLASSES)
# ============================================================

NEURONSPARK_ATTENTION_CLASSES = {
    "eager": NeuronSparkAttention,
    "flash_attention_2": NeuronSparkFlashAttention2,
    "sdpa": NeuronSparkSdpaAttention,
}


# ============================================================
# NeuronSparkBlock (renamed from NemotronHBlock)
# ============================================================

class NeuronSparkBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NeuronSparkRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # S: SSM, *: Attention, -: MLP, E: MoE
        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "ssm":
            self.mixer = BioSSMMixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NEURONSPARK_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NeuronSparkMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NeuronSparkMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid layer pattern {config.hybrid_override_pattern[layer_idx]}")

    def forward(
        self,
        hidden_states,
        cache_params: Optional[NeuronSparkDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
            # * Use torch.cuda.stream() to avoid NaN issues when using multiple GPUs
            residual = hidden_states

            if self.block_type == "ssm":
                # BioSSMLayer has internal Pre-LN, skip block-level norm
                hidden_states = self.mixer(hidden_states)
            else:
                hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                if self.block_type == "attention":
                    hidden_states = self.mixer(
                        hidden_states, cache_position=cache_position
                    )
                    hidden_states = hidden_states[0]
                elif self.block_type in ["mlp", "moe"]:
                    hidden_states = self.mixer(
                        hidden_states
                    )
                else:
                    raise ValueError(f"Invalid block_type: {self.block_type}")

            hidden_states = residual + hidden_states
            return hidden_states


# ============================================================
# NeuronSparkPreTrainedModel (renamed from NemotronHPreTrainedModel)
# ============================================================

# Copied from transformers.models.mamba.modeling_mamba2.Mamba2PreTrainedModel
class NeuronSparkPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NeuronSparkConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["NeuronSparkBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, BioSSMMixer):
            # BioSSMLayer does its own init internally, nothing to do here
            pass

        if isinstance(module, NeuronSparkTopkRouter):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        # TODO: Check
        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/sqrt(N) where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if getattr(p, "_is_hf_initialized", False):
                    continue
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


# ============================================================
# Output dataclasses
# ============================================================

@dataclass
class NeuronSparkOutput(ModelOutput):
    """
    Class for the NeuronSpark model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`NeuronSparkDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        ponder_cost (`torch.FloatTensor`):
            Total ponder cost from all SSM layers.
        ek_floor_cost (`torch.FloatTensor`):
            Total E[K] floor penalty cost from all SSM layers.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[NeuronSparkDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ponder_cost: Optional[torch.FloatTensor] = None
    ek_floor_cost: Optional[torch.FloatTensor] = None


@dataclass
class NeuronSparkCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`NeuronSparkDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        ponder_cost (`torch.FloatTensor`):
            Total ponder cost from all SSM layers.
        ek_floor_cost (`torch.FloatTensor`):
            Total E[K] floor penalty cost from all SSM layers.
        mtp_loss (`torch.FloatTensor`):
            Multi-Token Prediction auxiliary loss.
        last_loss (`torch.FloatTensor`):
            Alias for loss (backward compat with training script).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[NeuronSparkDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ponder_cost: Optional[torch.FloatTensor] = None
    ek_floor_cost: Optional[torch.FloatTensor] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    last_loss: Optional[torch.FloatTensor] = None


# ============================================================
# Docstrings
# ============================================================

NEURONSPARK_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`NeuronSparkConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

NEURONSPARK_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            If `cache_params.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        position_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        cache_params (`NeuronSparkDynamicCache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `cache_params` is passed, `cache_position` should also be passed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
"""


# ============================================================
# NeuronSparkModel (renamed from NemotronHModel)
# ============================================================

@add_start_docstrings(
    "The bare NeuronSpark Model transformer outputting raw hidden-states without any specific head on top.",
    NEURONSPARK_START_DOCSTRING,
)
class NeuronSparkModel(NeuronSparkPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NeuronSparkBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = NeuronSparkRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in list(state_dict.keys()):
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def _reset_ssm_states(self):
        """重置所有 SSM 层的 PLIF 神经元膜电位状态。"""
        for layer_block in self.layers:
            if layer_block.block_type == "ssm":
                bio_ssm = layer_block.mixer.bio_ssm
                bio_ssm.input_neuron.v = 0.
                bio_ssm.snn_block.hidden_neuron.v = 0.

    @add_start_docstrings_to_model_forward(NEURONSPARK_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=NeuronSparkOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[NeuronSparkDynamicCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, NeuronSparkOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # From zamba_modeling.py
        if use_cache and cache_params is None:
            logger.warning_once(
                "NeuronSpark requires an initialized `NeuronSparkDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )

        hidden_states = inputs_embeds

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Reset SSM neuron states at the start of each forward pass
        self._reset_ssm_states()

        # Collect SSM costs
        total_ponder_cost = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        total_ek_floor_cost = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        ssm_count = 0

        for layer_idx, mixer_block in enumerate(self.layers):
            # Depending on the layer type we opt for 2D base attention mask (SSM) or 4D causal mask (Attention)
            if mixer_block.block_type == "ssm":
                layer_mask = mamba_mask
            elif mixer_block.block_type == "attention":
                layer_mask = causal_mask
            elif mixer_block.block_type in ["mlp", "moe"]:
                layer_mask = None
            else:
                raise ValueError(f"Invalid block_type: {mixer_block.block_type}")

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, layer_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )

            # Collect ponder_cost and ek_floor_cost from SSM layers
            if mixer_block.block_type == "ssm":
                if mixer_block.mixer.ponder_cost is not None:
                    total_ponder_cost = total_ponder_cost + mixer_block.mixer.ponder_cost
                if mixer_block.mixer.ek_floor_cost is not None:
                    total_ek_floor_cost = total_ek_floor_cost + mixer_block.mixer.ek_floor_cost
                ssm_count += 1

        hidden_states = self.norm_f(hidden_states)

        # Average costs over SSM layers
        if ssm_count > 0:
            total_ponder_cost = total_ponder_cost / ssm_count
            total_ek_floor_cost = total_ek_floor_cost / ssm_count

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states, total_ponder_cost, total_ek_floor_cost] if v is not None)

        return NeuronSparkOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            ponder_cost=total_ponder_cost,
            ek_floor_cost=total_ek_floor_cost,
        )

    # Copied from transformers.models.jamba.modeling_jamba.JambaModel._update_causal_mask
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask


# ============================================================
# NeuronSparkForCausalLM (renamed from NemotronHForCausalLM)
# ============================================================

@add_start_docstrings(
    """
    The NeuronSpark Model transformer with a language modeling head on top (linear layer with weights not tied to the input
    embeddings).
    """,
    NEURONSPARK_START_DOCSTRING,
)
class NeuronSparkForCausalLM(NeuronSparkPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = NeuronSparkModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_head = MTPHead(
            D=config.hidden_size,
            vocab_size=config.vocab_size,
            n_heads=config.n_mtp_heads,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.backbone

    def set_decoder(self, decoder):
        self.backbone = decoder

    def compensate_modulation_gradients(self):
        """调制参数梯度补偿: 对 BioSSMLayer 的 b_beta, b_alpha, b_th 施加梯度乘数。"""
        for layer_block in self.backbone.layers:
            if layer_block.block_type == "ssm":
                snn_block = layer_block.mixer.bio_ssm.snn_block
                for name in ['b_beta', 'b_alpha', 'b_th']:
                    param = getattr(snn_block, name)
                    if param.grad is not None:
                        param.grad.data.mul_(10.0)

    def get_param_groups(self, weight_decay=0.01, neuron_lr_mult=10.0):
        """参数分组: 返回 dict[str, list[Parameter]]，匹配 train_fsdp_v3.py 的分组逻辑。

        Keys:
            input_neurons: PLIF 输入神经元 v_th, beta
            ssm_bias: BioSSM 调制偏置 b_beta, b_alpha, b_th
            rms_norms: 所有 RMSNorm weight
            halt_projs: PonderNet halt_proj 参数
            embedding: embedding weight
            moe_router: MoE router weight + e_score_correction_bias
            moe_norms: (未使用，保留兼容)
            attn_norms: (未使用，保留兼容)
            decay: 其余 weight decay 参数
        """
        groups = {
            'input_neurons': [],
            'ssm_bias': [],
            'rms_norms': [],
            'halt_projs': [],
            'embedding': [],
            'moe_router': [],
            'moe_norms': [],
            'attn_norms': [],
            'decay': [],
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'input_neuron.v_th' in name or 'input_neuron.beta' in name:
                groups['input_neurons'].append(param)
            elif any(k in name for k in ['b_beta', 'b_alpha', 'b_th']):
                groups['ssm_bias'].append(param)
            elif 'halt_proj' in name:
                groups['halt_projs'].append(param)
            elif 'embeddings.weight' in name:
                groups['embedding'].append(param)
            elif 'gate.weight' in name or 'e_score_correction_bias' in name:
                groups['moe_router'].append(param)
            elif 'norm' in name.lower():
                groups['rms_norms'].append(param)
            else:
                groups['decay'].append(param)

        return groups

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`
        empty_past_kv = past_key_values is None

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if not empty_past_kv:
            if (
                inputs_embeds is not None  # Exception 1
                or cache_position[-1] >= input_ids.shape[1]  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = NeuronSparkDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(NEURONSPARK_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=NeuronSparkCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        target_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[NeuronSparkDynamicCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, NeuronSparkCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        target_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Target token IDs for computing loss + MTP loss. If provided, acts as labels.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # target_ids 兼容: 同时支持 labels 和 target_ids
        if labels is None and target_ids is not None:
            labels = target_ids

        backbone_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_outputs[0]
        ponder_cost = backbone_outputs.ponder_cost if return_dict else torch.tensor(0.0, device=hidden_states.device)
        ek_floor_cost = backbone_outputs.ek_floor_cost if return_dict else torch.tensor(0.0, device=hidden_states.device)

        # TODO: Check zamba_modeling.py
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        mtp_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # MTP loss
            mtp_loss = self.mtp_head(
                h=hidden_states,
                target_ids=labels,
                lm_head_weight=self.lm_head.weight,
                embed_weight=self.backbone.embeddings.weight,
            )

        if not return_dict:
            output = (logits,) + backbone_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return NeuronSparkCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=backbone_outputs.cache_params,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
            ponder_cost=ponder_cost,
            ek_floor_cost=ek_floor_cost,
            mtp_loss=mtp_loss,
            last_loss=loss,
        )


# ============================================================
# Backward-compat aliases
# ============================================================

SparkMLP = NeuronSparkMLP
SparkMOE = NeuronSparkMOE
SparkTopkRouter = NeuronSparkTopkRouter
SparkAttention = NeuronSparkAttention
SparkBlock = NeuronSparkBlock
NeuronSparkV3Config = NeuronSparkConfig
NeuronSparkV3ForCausalLM = NeuronSparkForCausalLM
