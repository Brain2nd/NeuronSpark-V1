"""
v3 蒸馏封装: NVIDIA NemotronH + 并行 BioSSM

核心设计:
  1. 加载 NemotronHForCausalLM (pretrained, 全部冻结)
  2. 在每个 Mamba 位置并行挂载 BioSSMMixer (可训练)
  3. Forward: Mamba(no_grad) 和 BioSSM 并行跑, BioSSM 输出进入残差流
  4. 每个 Mamba 位置收集 cosine alignment loss
  5. 最终 logits 走 student 路径 (BioSSM)

显存优化:
  - 非 SSM 层冻结, 无 optimizer states
  - FSDP use_orig_params=True, 分片所有参数
  - gradient checkpointing 可选开启
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from atomic_ops.bio_ssm_layer import BioSSMLayer


# ============================================================
# BioSSM 配置 (独立于 NeuronSparkConfig, 只需 SSM 相关字段)
# ============================================================

@dataclass
class BioSSMConfig:
    """BioSSMLayer 蒸馏配置。"""
    hidden_size: int = 2688
    ssm_N: int = 4
    ssm_K: int = 16
    ssm_v_th_min: float = 0.1
    ssm_ek_floor: float = 4.0
    num_hidden_layers: int = 52


# ============================================================
# BioSSMMixer (从 model_v3.py 复制, 独立于 HF 依赖)
# ============================================================

class BioSSMMixer(nn.Module):
    """包装 BioSSMLayer 为 NVIDIA mixer 接口。

    BioSSMLayer 内部有 Pre-LN, 所以接收原始 hidden_states (不经 block norm)。
    输出是残差增量 (剥离了 BioSSMLayer 内部的 h + out)。
    """

    def __init__(self, config: BioSSMConfig, layer_idx: int):
        super().__init__()
        self.bio_ssm = BioSSMLayer(
            D=config.hidden_size,
            N=config.ssm_N,
            K=config.ssm_K,
            v_th_min=config.ssm_v_th_min,
            num_layers=config.num_hidden_layers,
            layer_idx=layer_idx,
            ek_floor=config.ssm_ek_floor,
        )
        self.ponder_cost = None
        self.ek_floor_cost = None

    def forward(self, hidden_states, **kwargs):
        """(batch, seq, D) → (batch, seq, D) 残差增量。"""
        h = hidden_states.permute(1, 0, 2).contiguous()  # (B,S,D) → (S,B,D)
        h_out, pc, efc = self.bio_ssm(h)
        out = h_out - h  # 剥离 BioSSMLayer 内部残差
        self.ponder_cost = pc
        self.ek_floor_cost = efc
        return out.permute(1, 0, 2).contiguous()  # (S,B,D) → (B,S,D)


# ============================================================
# 蒸馏输出
# ============================================================

@dataclass
class DistillOutput:
    """蒸馏 forward 输出。"""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    hidden_loss: Optional[torch.Tensor] = None
    ponder_cost: Optional[torch.Tensor] = None
    ek_floor_cost: Optional[torch.Tensor] = None


# ============================================================
# 蒸馏损失
# ============================================================

def compute_layer_cosine_loss(bio_out, mamba_out):
    """单层 cosine 距离 loss。"""
    return (1.0 - F.cosine_similarity(
        bio_out.reshape(-1, bio_out.shape[-1]),
        mamba_out.reshape(-1, mamba_out.shape[-1]),
        dim=-1,
    )).mean()


# ============================================================
# DistillHybridModel
# ============================================================

class DistillHybridModel(nn.Module):
    """NVIDIA NemotronH + 并行 BioSSM 蒸馏模型。

    Args:
        nvidia_model: NemotronHForCausalLM (pretrained, 将被冻结)
        bio_ssm_config: BioSSM 配置
    """

    def __init__(self, nvidia_model, bio_ssm_config: BioSSMConfig,
                 gradient_checkpointing: bool = True):
        super().__init__()
        self.nvidia_model = nvidia_model
        self.bio_ssm_config = bio_ssm_config
        self.gradient_checkpointing = gradient_checkpointing

        # 冻结所有原始参数
        for p in self.nvidia_model.parameters():
            p.requires_grad_(False)

        # 在每个 Mamba 位置挂载 BioSSMMixer
        self.bio_ssm_modules = nn.ModuleDict()
        self.mamba_indices = []

        for idx, block in enumerate(self.nvidia_model.backbone.layers):
            if block.block_type == "mamba":
                self.mamba_indices.append(idx)
                self.bio_ssm_modules[str(idx)] = BioSSMMixer(
                    bio_ssm_config, layer_idx=idx,
                )

        self._num_mamba = len(self.mamba_indices)
        self._mamba_set = set(self.mamba_indices)
        self._layer_cosine_dict = {}  # idx → cosine distance (供 dashboard 读取)

    def _reset_bio_ssm_states(self):
        """重置所有 BioSSM 层的 PLIF 膜电位。"""
        for bio_mixer in self.bio_ssm_modules.values():
            bio_mixer.bio_ssm.input_neuron.v = 0.
            bio_mixer.bio_ssm.snn_block.hidden_neuron.v = 0.

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DistillOutput:
        backbone = self.nvidia_model.backbone

        # Embedding
        hidden_states = backbone.embeddings(input_ids)
        batch, seq_len = input_ids.shape

        # Masks
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        causal_mask = backbone._update_causal_mask(attention_mask, hidden_states, cache_position)
        mamba_mask = backbone._update_mamba_mask(attention_mask, cache_position)

        # 重置 BioSSM 神经元状态
        self._reset_bio_ssm_states()

        layer_cosine_losses = []
        self._layer_cosine_dict = {}
        total_ponder = 0.0
        total_ek_floor = 0.0

        for idx, block in enumerate(backbone.layers):
            if idx in self._mamba_set:
                # ===== Mamba 位置: 并行 forward =====
                residual = hidden_states

                # Mamba 路径 (冻结, no_grad)
                # NemotronHBlock 的 forward 会先 norm 再传给 mixer
                normed = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
                with torch.no_grad():
                    mamba_out = block.mixer(
                        normed, cache_params=None, cache_position=cache_position,
                        attention_mask=mamba_mask,
                    )

                # BioSSM 路径 (可训练, 内部有自己的 norm)
                bio_mixer = self.bio_ssm_modules[str(idx)]
                bio_out = bio_mixer(hidden_states)

                # 逐层 cosine alignment loss
                cos_loss = compute_layer_cosine_loss(bio_out, mamba_out.detach())
                layer_cosine_losses.append(cos_loss)
                self._layer_cosine_dict[idx] = cos_loss.detach().item()

                # 收集 SNN cost
                if bio_mixer.ponder_cost is not None:
                    total_ponder = total_ponder + bio_mixer.ponder_cost
                if bio_mixer.ek_floor_cost is not None:
                    total_ek_floor = total_ek_floor + bio_mixer.ek_floor_cost

                # Student 残差流
                hidden_states = residual + bio_out
            else:
                # ===== 非 Mamba 层: 冻结直通 =====
                if block.block_type == "attention":
                    layer_mask = causal_mask
                elif block.block_type == "mamba":
                    layer_mask = mamba_mask  # 不应走到这里
                else:
                    layer_mask = None

                if self.gradient_checkpointing and self.training:
                    hidden_states = ckpt_fn(
                        block, hidden_states, None, cache_position, layer_mask,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        cache_params=None,
                        cache_position=cache_position,
                        attention_mask=layer_mask,
                    )

        # Final norm + LM head
        hidden_states = backbone.norm_f(hidden_states)
        logits = self.nvidia_model.lm_head(
            hidden_states.to(self.nvidia_model.lm_head.weight.dtype)
        ).float()

        # Loss
        ce_loss = None
        hidden_loss = None
        loss = None

        if labels is not None:
            # CE loss
            # PretrainDataset 已做 shift: X=input_id[:-1], Y=input_id[1:]
            # logits[i] 预测 labels[i], 无需再 shift
            if loss_mask is not None:
                mask = loss_mask.contiguous().view(-1).float()
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='none',
                )
                ce_loss = (ce_loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            # Hidden alignment loss (平均逐层 cosine)
            if layer_cosine_losses:
                hidden_loss = sum(layer_cosine_losses) / len(layer_cosine_losses)

        # Ponder / EK floor (平均)
        ponder_cost = total_ponder / max(self._num_mamba, 1)
        ek_floor_cost = total_ek_floor / max(self._num_mamba, 1)

        return DistillOutput(
            loss=loss,
            logits=logits,
            ce_loss=ce_loss,
            hidden_loss=hidden_loss,
            ponder_cost=ponder_cost,
            ek_floor_cost=ek_floor_cost,
        )

    def compensate_modulation_gradients(self):
        """调制参数梯度补偿。"""
        for bio_mixer in self.bio_ssm_modules.values():
            snn_block = bio_mixer.bio_ssm.snn_block
            for name in ['b_beta', 'b_alpha', 'b_th']:
                param = getattr(snn_block, name)
                if param.grad is not None:
                    param.grad.data.mul_(10.0)

    def get_bio_ssm_param_groups(self, lr=2e-4, neuron_lr_mult=10.0, weight_decay=0.1):
        """只收集 BioSSM 的可训练参数, 分 decay/no_decay/neuron 三组。"""
        decay, no_decay, neuron = [], [], []
        for name, p in self.bio_ssm_modules.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ['b_beta', 'b_alpha', 'b_th',
                                        'input_neuron.v_th', 'input_neuron.w']):
                neuron.append(p)
            elif 'norm' in name.lower() or 'halt_proj' in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {'params': decay, 'lr': lr, 'weight_decay': weight_decay, 'lr_mult': 1.0},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0.0, 'lr_mult': 1.0},
            {'params': neuron, 'lr': lr * neuron_lr_mult, 'weight_decay': 0.0,
             'lr_mult': float(neuron_lr_mult)},
        ]

    def save_bio_ssm_state(self):
        """提取仅 BioSSM 模块的 state_dict (不含冻结的 NVIDIA 模型)。"""
        return {k: v.cpu() for k, v in self.bio_ssm_modules.state_dict().items()}

    def load_bio_ssm_state(self, state_dict):
        """加载 BioSSM 模块的 state_dict。"""
        self.bio_ssm_modules.load_state_dict(state_dict)
