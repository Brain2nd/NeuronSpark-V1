"""
v3 蒸馏封装: NVIDIA NemotronH + 并行 BioSSM

核心设计:
  1. 加载 NemotronHForCausalLM (pretrained, 全部冻结)
  2. 在每个 Mamba 位置并行挂载 BioSSMMixer (可训练)
  3. Forward: Mamba(no_grad) 和 BioSSM 并行跑, BioSSM 输出进入残差流
  4. 每个 Mamba 位置收集 cosine alignment loss
  5. 最终 logits 走 student 路径 (BioSSM), CE loss 梯度穿越冻结层回传

通讯优化 (layer-batched accumulation):
  - 将梯度累积从 micro-step 维度翻转到 layer 维度
  - 冻结层: accum 个 micro-batch 拼接 batch 维, 一次 FSDP all-gather
  - BioSSM: 逐 micro-batch 处理 (控制显存峰值)
  - 冻结层 all-gather 从 52×accum 降到 52×1 (accum=16 时 -93.75%)

显存优化:
  - 全层 gradient checkpointing: 冻结层和 BioSSM 均不保存中间激活, backward 时重算
    (冻结层激活省数 GB, BioSSM 每层 seq=2048 时 ~5.4GB → checkpoint 后 ~11MB)
  - FSDP use_orig_params=True, 分片所有参数
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

    梯度流: CE loss → norm_f → 冻结层(保持梯度流) → BioSSM, 端到端训练。
    冻结层参数 requires_grad=False, 不产生梯度更新, 但梯度可穿越。
    """

    def __init__(self, nvidia_model, bio_ssm_config: BioSSMConfig):
        super().__init__()
        self.nvidia_model = nvidia_model
        self.bio_ssm_config = bio_ssm_config

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

        # 在 Mamba mixer 上注册 forward hook, 捕获 mixer 输出用于 cosine alignment
        # hook 在 block() 内部触发, 此时 FSDP 已 all-gather 参数, 无需直接访问子组件
        self._mamba_mixer_outputs = {}
        self._mixer_hooks = []
        for idx in self.mamba_indices:
            block = self.nvidia_model.backbone.layers[idx]
            hook = block.mixer.register_forward_hook(self._make_mixer_hook(idx))
            self._mixer_hooks.append(hook)

    def _make_mixer_hook(self, idx):
        """创建 Mamba mixer forward hook, 捕获输出用于 cosine alignment。"""
        def hook(module, input, output):
            self._mamba_mixer_outputs[idx] = output
        return hook

    @staticmethod
    def _make_frozen_fn(block, cache_position, layer_mask):
        """创建冻结层 forward 闭包 (正确捕获循环变量, 避免 late binding)。"""
        def fn(hidden_states):
            return block(hidden_states, cache_params=None,
                         cache_position=cache_position,
                         attention_mask=layer_mask)
        return fn

    @staticmethod
    def _make_ce_fn(norm_f, lm_head, lm_dtype, loss_mask_chunk):
        """创建 CE loss 分块计算闭包 (gradient checkpoint 用)。

        每次只算 1 个 micro-batch 的 logits, 避免全量 logits 占数 GB 显存。
        """
        def fn(h_c, l_c):
            h_norm = norm_f(h_c)
            logits = lm_head(h_norm.to(lm_dtype)).float()
            V = logits.size(-1)
            if loss_mask_chunk is not None:
                mask = loss_mask_chunk.contiguous().view(-1).float()
                ce = F.cross_entropy(
                    logits.view(-1, V), l_c.view(-1), reduction='none')
                return (ce * mask).sum(), mask.sum()
            else:
                valid = (l_c.view(-1) != -100).float()
                ce = F.cross_entropy(
                    logits.view(-1, V), l_c.view(-1),
                    ignore_index=-100, reduction='none')
                return (ce * valid).sum(), valid.sum()
        return fn

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
        accumulation_steps: int = 1,
    ) -> DistillOutput:
        """蒸馏前向传播。

        当 accumulation_steps > 1 时, input_ids 的 batch 维包含 accum 个 micro-batch
        拼接. 冻结层一次性处理拼接 batch (1 次 FSDP all-gather), BioSSM 逐 micro-batch
        处理 (控制显存峰值). 通讯量从 52×accum 降到 52.

        Args:
            input_ids: (accum * batch, seq) — 多个 micro-batch 在 batch 维拼接
            labels: (accum * batch, seq) 或 None
            loss_mask: (accum * batch, seq) 或 None
            attention_mask: 通常 None (训练模式)
            accumulation_steps: micro-batch 数量, 1 = 无累积 (兼容旧行为)
        """
        backbone = self.nvidia_model.backbone
        accum = accumulation_steps

        # Embedding
        hidden_states = backbone.embeddings(input_ids)
        batch_total, seq_len = input_ids.shape

        # Masks (基于完整拼接 batch 创建, batch 维独立, 安全)
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        causal_mask = backbone._update_causal_mask(
            attention_mask, hidden_states, cache_position)
        mamba_mask = backbone._update_mamba_mask(attention_mask, cache_position)

        layer_cosine_losses = []
        self._layer_cosine_dict = {}
        total_ponder = torch.tensor(0.0, device=hidden_states.device)
        total_ek_floor = torch.tensor(0.0, device=hidden_states.device)
        self._mamba_mixer_outputs = {}

        for idx, block in enumerate(backbone.layers):
            if idx in self._mamba_set:
                # ===== Mamba 位置 =====
                # Teacher: 拼接 batch 一次 forward (1 次 FSDP all-gather)
                with torch.no_grad():
                    block(
                        hidden_states,
                        cache_params=None,
                        cache_position=cache_position,
                        attention_mask=mamba_mask,
                    )
                mamba_out = self._mamba_mixer_outputs.pop(idx)

                # BioSSM: 逐 micro-batch (控制显存, K=16 展开占大量显存)
                bio_mixer = self.bio_ssm_modules[str(idx)]
                h_chunks = hidden_states.chunk(accum, dim=0)
                m_chunks = mamba_out.chunk(accum, dim=0)
                del mamba_out

                bio_outs = []
                layer_cos_sum = 0.0
                for i in range(accum):
                    # 每个 micro-batch 重置神经元膜电位
                    bio_mixer.bio_ssm.input_neuron.v = 0.
                    bio_mixer.bio_ssm.snn_block.hidden_neuron.v = 0.

                    if self.training:
                        bo = ckpt_fn(bio_mixer, h_chunks[i],
                                     use_reentrant=False)
                    else:
                        bo = bio_mixer(h_chunks[i])
                    bio_outs.append(bo)

                    cos_loss = compute_layer_cosine_loss(
                        bo, m_chunks[i].detach())
                    layer_cosine_losses.append(cos_loss)
                    layer_cos_sum += cos_loss.detach().item()

                    if bio_mixer.ponder_cost is not None:
                        total_ponder = total_ponder + bio_mixer.ponder_cost
                    if bio_mixer.ek_floor_cost is not None:
                        total_ek_floor = total_ek_floor + bio_mixer.ek_floor_cost

                self._layer_cosine_dict[idx] = layer_cos_sum / accum

                # 重组 bio_out 到拼接 batch 维
                if accum > 1:
                    bio_out_cat = torch.cat(bio_outs, dim=0)
                else:
                    bio_out_cat = bio_outs[0]
                hidden_states = hidden_states + bio_out_cat
                del h_chunks, m_chunks, bio_outs, bio_out_cat

            else:
                # ===== 冻结层: gradient checkpointing 省激活显存 =====
                # 不保存中间激活, backward 时重算; 梯度仍穿越到 BioSSM
                if block.block_type == "attention":
                    layer_mask = causal_mask
                else:
                    layer_mask = None

                if self.training:
                    _fn = self._make_frozen_fn(block, cache_position, layer_mask)
                    hidden_states = ckpt_fn(_fn, hidden_states,
                                            use_reentrant=False)
                else:
                    hidden_states = block(
                        hidden_states,
                        cache_params=None,
                        cache_position=cache_position,
                        attention_mask=layer_mask,
                    )

        # CE loss: 逐 micro-batch 分块计算, 避免全量 logits OOM
        # (vocab 维度大, accum 个 micro-batch 拼接后 logits 在 float32 下数 GB)
        ce_loss = None
        logits = None
        if labels is not None:
            lm_head = self.nvidia_model.lm_head
            lm_dtype = lm_head.weight.dtype
            h_chunks = hidden_states.chunk(accum, dim=0)
            l_chunks = labels.chunk(accum, dim=0)
            m_chunks = (loss_mask.chunk(accum, dim=0)
                        if loss_mask is not None else [None] * accum)

            ce_sum = torch.tensor(0.0, device=hidden_states.device)
            mask_sum = torch.tensor(0.0, device=hidden_states.device)

            for h_c, l_c, m_c in zip(h_chunks, l_chunks, m_chunks):
                # 每块 checkpoint: backward 时重算 logits, 不保存全量
                _fn = self._make_ce_fn(backbone.norm_f, lm_head, lm_dtype, m_c)
                chunk_ce, chunk_mask = ckpt_fn(
                    _fn, h_c, l_c, use_reentrant=False)
                ce_sum = ce_sum + chunk_ce
                mask_sum = mask_sum + chunk_mask

            ce_loss = ce_sum / mask_sum.clamp(min=1)

        # Hidden alignment loss (平均: 所有层 × 所有 micro-batch)
        hidden_loss = None
        if layer_cosine_losses:
            hidden_loss = sum(layer_cosine_losses) / len(layer_cosine_losses)

        # Ponder / EK floor (平均: 所有层 × 所有 micro-batch)
        n_total = max(self._num_mamba * accum, 1)
        ponder_cost = total_ponder / n_total
        ek_floor_cost = total_ek_floor / n_total

        return DistillOutput(
            loss=None,
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
