"""
v3 渐进蒸馏: Mamba → BioSSM (back-to-front)

两阶段设计:
  Stage 1 — 逐块对齐: 纯 teacher forward, 捕获 (block_input, mamba_output),
            单块 BioSSM 训练. MSE + cosine + ponder + ek_floor.
            零残差流污染, 零 CE 梯度消失.
  Stage 2 — 端到端整合: 渐进解冻 BioSSM 块, dual forward (teacher + student),
            KL + CE + MSE + ponder. 温度线性衰减.

复用 distill_wrapper_v3.py 的: BioSSMMixer, BioSSMConfig, compensate, clip, param_groups.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from atomic_ops.bio_ssm_layer import BioSSMLayer
from distill_wrapper_v3 import BioSSMMixer, BioSSMConfig


# ============================================================
# 输出 dataclasses
# ============================================================

@dataclass
class Stage1Output:
    """Stage 1 逐块对齐输出。"""
    mse_loss: torch.Tensor
    cos_loss: torch.Tensor
    ponder_cost: torch.Tensor
    ek_floor_cost: torch.Tensor


@dataclass
class Stage2Output:
    """Stage 2 端到端整合输出。"""
    kl_loss: torch.Tensor
    ce_loss: torch.Tensor
    mse_loss: torch.Tensor
    ponder_cost: torch.Tensor
    ek_floor_cost: torch.Tensor
    per_block_cosine: Dict[int, float] = field(default_factory=dict)


# ============================================================
# ProgressiveDistillModel
# ============================================================

class ProgressiveDistillModel(nn.Module):
    """渐进式蒸馏模型: Stage 1 逐块 + Stage 2 端到端。

    Args:
        nvidia_model: NemotronHForCausalLM (pretrained, 将被冻结)
        bio_ssm_config: BioSSM 配置
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

        # Stage 状态
        self._stage_mode = None             # 'stage1' 或 'stage2'
        self._active_block_idx = None       # Stage 1: 当前训练的块索引
        self._active_stage2_set = set()     # Stage 2: 活跃块集合
        self._replaced_blocks = set()       # Stage 2: 已替换为 BioSSM 的块

        # Hook 存储
        self._block_inputs = {}             # idx → hidden_states (block 输入)
        self._mamba_mixer_outputs = {}      # idx → mixer output
        self._hooks = []

        # 注册 hooks
        for idx in self.mamba_indices:
            block = self.nvidia_model.backbone.layers[idx]
            # Pre-hook: 捕获 block 输入
            pre_h = block.register_forward_pre_hook(self._make_pre_hook(idx))
            self._hooks.append(pre_h)
            # Post-hook on mixer: 捕获 Mamba mixer 输出
            post_h = block.mixer.register_forward_hook(self._make_mixer_hook(idx))
            self._hooks.append(post_h)

        # E[K] EMA (复用 distill_wrapper_v3 逻辑)
        self._ek_ema = {}

    def _make_pre_hook(self, idx):
        """block 级 pre-hook: 捕获输入 hidden_states。

        Stage 1: 只捕获活跃块 (block_input 用于 BioSSM 的输入)
        Stage 2: 不需要 block_input (student 有自己的残差流)
        """
        def hook(module, args):
            if self._stage_mode == 'stage1' and idx == self._active_block_idx:
                self._block_inputs[idx] = args[0].detach()
        return hook

    def _make_mixer_hook(self, idx):
        """mixer 级 post-hook: 捕获 Mamba mixer 输出。

        Stage 1: 只捕获活跃块
        Stage 2: 全部捕获 (用于逐块 MSE alignment)
        """
        def hook(module, input, output):
            if self._stage_mode == 'stage1' and idx != self._active_block_idx:
                return
            self._mamba_mixer_outputs[idx] = output.detach()
        return hook

    # ====== forward 路由 (FSDP 必须通过此入口) ======

    def forward(self, input_ids, labels=None, loss_mask=None, accumulation_steps=1):
        """统一入口: FSDP 通过此方法调用, 触发顶层参数 all-gather。

        路由到 stage1 或 stage2, 由 set_stage1_block / set_stage2_active_blocks 设定。
        """
        if self._stage_mode == 'stage1':
            return self.forward_stage1(input_ids, accumulation_steps)
        elif self._stage_mode == 'stage2':
            return self.forward_stage2(input_ids, labels, loss_mask, accumulation_steps)
        else:
            raise RuntimeError(
                "请先调用 set_stage1_block() 或 set_stage2_active_blocks() 设置模式")

    # ====== 膜电位重置 ======

    def _reset_bio_ssm_states(self, idx=None):
        """重置 BioSSM 层的 PLIF 膜电位。idx=None 重置全部。"""
        if idx is not None:
            bio_mixer = self.bio_ssm_modules[str(idx)]
            bio_mixer.bio_ssm.input_neuron.v = 0.
            bio_mixer.bio_ssm.snn_block.hidden_neuron.v = 0.
        else:
            for bio_mixer in self.bio_ssm_modules.values():
                bio_mixer.bio_ssm.input_neuron.v = 0.
                bio_mixer.bio_ssm.snn_block.hidden_neuron.v = 0.

    # ====== Stage 1: 逐块对齐 ======

    def set_stage1_block(self, mamba_idx: int):
        """冻结所有 BioSSM, 仅解冻 mamba_idx 对应的块。"""
        assert mamba_idx in self._mamba_set, f"idx {mamba_idx} 不是 Mamba 层"
        self._stage_mode = 'stage1'
        self._active_block_idx = mamba_idx

        # 冻结所有 BioSSM
        for idx_str, bio_mixer in self.bio_ssm_modules.items():
            requires_grad = (int(idx_str) == mamba_idx)
            for p in bio_mixer.parameters():
                p.requires_grad_(requires_grad)

    def forward_stage1(self, input_ids, accumulation_steps=1):
        """Stage 1: 纯 teacher forward + 单块 BioSSM 训练。

        Args:
            input_ids: (accum * batch, seq)
            accumulation_steps: micro-batch 数量

        Returns:
            Stage1Output
        """
        idx = self._active_block_idx
        assert idx is not None, "请先调用 set_stage1_block()"
        backbone = self.nvidia_model.backbone
        accum = accumulation_steps
        bio_mixer = self.bio_ssm_modules[str(idx)]

        # 清空 hook 缓存
        self._block_inputs = {}
        self._mamba_mixer_outputs = {}

        # ===== 1. Teacher full forward (no_grad) =====
        # Hook 会自动捕获 block_input 和 mamba_mixer_output
        with torch.no_grad():
            hidden_states = backbone.embeddings(input_ids)
            batch_total, seq_len = input_ids.shape
            cache_position = torch.arange(seq_len, device=hidden_states.device)
            causal_mask = backbone._update_causal_mask(
                None, hidden_states, cache_position)
            mamba_mask = backbone._update_mamba_mask(None, cache_position)

            for i, block in enumerate(backbone.layers):
                if block.block_type == "attention":
                    layer_mask = causal_mask
                elif block.block_type == "mamba":
                    layer_mask = mamba_mask
                else:
                    layer_mask = None
                hidden_states = block(
                    hidden_states, cache_params=None,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )

        # ===== 2. 提取目标 (pop 释放引用, 减少显存占用) =====
        block_input = self._block_inputs.pop(idx, None)
        mamba_out = self._mamba_mixer_outputs.pop(idx, None)
        assert block_input is not None, f"未捕获到 block {idx} 的输入"
        assert mamba_out is not None, f"未捕获到 block {idx} 的 Mamba 输出"

        # ===== 3. BioSSM forward (with grad), 逐 micro-batch =====
        h_chunks = block_input.chunk(accum, dim=0)
        m_chunks = mamba_out.chunk(accum, dim=0)
        del block_input, mamba_out

        total_mse = torch.tensor(0.0, device=input_ids.device)
        total_cos = torch.tensor(0.0, device=input_ids.device)
        total_ponder = torch.tensor(0.0, device=input_ids.device)
        total_ek_floor = torch.tensor(0.0, device=input_ids.device)

        for i in range(accum):
            self._reset_bio_ssm_states(idx)
            bio_out = ckpt_fn(bio_mixer, h_chunks[i], use_reentrant=False)

            # MSE loss
            mse = F.mse_loss(bio_out, m_chunks[i])
            total_mse = total_mse + mse

            # Cosine loss
            cos = (1.0 - F.cosine_similarity(
                bio_out.reshape(-1, bio_out.shape[-1]),
                m_chunks[i].reshape(-1, m_chunks[i].shape[-1]),
                dim=-1,
            )).mean()
            total_cos = total_cos + cos

            # PonderNet costs
            if bio_mixer.ponder_cost is not None:
                total_ponder = total_ponder + bio_mixer.ponder_cost
            if bio_mixer.ek_floor_cost is not None:
                total_ek_floor = total_ek_floor + bio_mixer.ek_floor_cost

        n = max(accum, 1)
        return Stage1Output(
            mse_loss=total_mse / n,
            cos_loss=total_cos / n,
            ponder_cost=total_ponder / n,
            ek_floor_cost=total_ek_floor / n,
        )

    # ====== Stage 2: 端到端整合 ======

    def set_stage2_active_blocks(self, active_indices: list):
        """设置 Stage 2 可训练块集合。

        active_indices 中的块: requires_grad=True
        其余 BioSSM: requires_grad=False (但仍参与 forward)
        """
        self._stage_mode = 'stage2'
        self._active_stage2_set = set(active_indices)
        self._replaced_blocks = set(self.mamba_indices)  # Stage 2 所有 Mamba 都用 BioSSM

        for idx_str, bio_mixer in self.bio_ssm_modules.items():
            is_active = int(idx_str) in self._active_stage2_set
            for p in bio_mixer.parameters():
                p.requires_grad_(is_active)

    @staticmethod
    def _make_frozen_fn(block, cache_position, layer_mask):
        """冻结层 forward 闭包 (gradient checkpoint 用)。"""
        def fn(hidden_states):
            return block(hidden_states, cache_params=None,
                         cache_position=cache_position,
                         attention_mask=layer_mask)
        return fn

    @staticmethod
    def _make_kl_ce_fn(norm_f, lm_head, lm_dtype, loss_mask_chunk, T):
        """KL + CE 分块计算闭包 (gradient checkpoint 用)。

        避免 vocab=131K 全量 logits 留在 autograd graph:
        accum=16 时可省 ~4GB (16 × 256MB/chunk)。
        """
        def fn(h_c, t_c, l_c):
            s_logits = lm_head(norm_f(h_c).to(lm_dtype)).float()
            with torch.no_grad():
                t_logits = lm_head(norm_f(t_c).to(lm_dtype)).float()

            V = s_logits.size(-1)
            mask = loss_mask_chunk.contiguous().view(-1).float()

            # CE
            ce = F.cross_entropy(s_logits.view(-1, V), l_c.view(-1), reduction='none')
            ce_val = (ce * mask).sum()

            # KL
            s_soft = F.log_softmax(s_logits.view(-1, V) / T, dim=-1)
            t_soft = F.softmax(t_logits.view(-1, V) / T, dim=-1)
            kl = F.kl_div(s_soft, t_soft, reduction='none').sum(-1)
            kl_val = (kl * mask).sum() * (T ** 2)

            mask_val = mask.sum()
            return ce_val, kl_val, mask_val
        return fn

    def forward_teacher_logits(self, input_ids):
        """纯 teacher forward → teacher final hidden (no_grad)。

        Returns:
            teacher_hidden_final: (batch, seq, D)
        """
        backbone = self.nvidia_model.backbone
        with torch.no_grad():
            hidden_states = backbone.embeddings(input_ids)
            seq_len = input_ids.shape[1]
            cache_position = torch.arange(seq_len, device=hidden_states.device)
            causal_mask = backbone._update_causal_mask(
                None, hidden_states, cache_position)
            mamba_mask = backbone._update_mamba_mask(None, cache_position)

            for block in backbone.layers:
                if block.block_type == "attention":
                    layer_mask = causal_mask
                elif block.block_type == "mamba":
                    layer_mask = mamba_mask
                else:
                    layer_mask = None
                hidden_states = block(
                    hidden_states, cache_params=None,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )
            return hidden_states

    def forward_stage2(self, input_ids, labels, loss_mask, accumulation_steps=1):
        """Stage 2: dual forward (teacher + student).

        Args:
            input_ids: (accum * batch, seq)
            labels: (accum * batch, seq)
            loss_mask: (accum * batch, seq)
            accumulation_steps: micro-batch 数量

        Returns:
            Stage2Output
        """
        backbone = self.nvidia_model.backbone
        accum = accumulation_steps

        # ===== Forward A: Teacher (no_grad) → teacher_hidden_final =====
        # 也捕获每块 mamba mixer 输出 (用于逐块 MSE)
        self._block_inputs = {}
        self._mamba_mixer_outputs = {}

        with torch.no_grad():
            teacher_h = backbone.embeddings(input_ids)
            batch_total, seq_len = input_ids.shape
            cache_position = torch.arange(seq_len, device=input_ids.device)
            causal_mask = backbone._update_causal_mask(
                None, teacher_h, cache_position)
            mamba_mask = backbone._update_mamba_mask(None, cache_position)

            for block in backbone.layers:
                if block.block_type == "attention":
                    layer_mask = causal_mask
                elif block.block_type == "mamba":
                    layer_mask = mamba_mask
                else:
                    layer_mask = None
                teacher_h = block(
                    teacher_h, cache_params=None,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )
            teacher_hidden_final = teacher_h.detach()

        # ===== Forward B: Student (BioSSM 替换 Mamba, with grad) =====
        hidden_states = backbone.embeddings(input_ids)

        per_block_cosine = {}
        total_mse = torch.tensor(0.0, device=input_ids.device)
        total_ponder = torch.tensor(0.0, device=input_ids.device)
        total_ek_floor = torch.tensor(0.0, device=input_ids.device)
        mse_count = 0

        for i, block in enumerate(backbone.layers):
            if i in self._replaced_blocks:
                bio_mixer = self.bio_ssm_modules[str(i)]
                mamba_out_target = self._mamba_mixer_outputs.get(i)

                # BioSSM 替换 Mamba: 逐 micro-batch
                h_chunks = hidden_states.chunk(accum, dim=0)
                bio_outs = []
                chunk_cos_sum = 0.0

                for c in range(accum):
                    self._reset_bio_ssm_states(i)
                    bo = ckpt_fn(bio_mixer, h_chunks[c], use_reentrant=False)
                    bio_outs.append(bo)

                    # 逐块 MSE + cosine (与 teacher mamba output 对比)
                    if mamba_out_target is not None:
                        m_chunk = mamba_out_target.chunk(accum, dim=0)[c]
                        mse = F.mse_loss(bo, m_chunk)
                        total_mse = total_mse + mse
                        mse_count += 1

                        with torch.no_grad():
                            cos_val = F.cosine_similarity(
                                bo.reshape(-1, bo.shape[-1]),
                                m_chunk.reshape(-1, m_chunk.shape[-1]),
                                dim=-1,
                            ).mean().item()
                            chunk_cos_sum += cos_val

                    if bio_mixer.ponder_cost is not None:
                        total_ponder = total_ponder + bio_mixer.ponder_cost
                    if bio_mixer.ek_floor_cost is not None:
                        total_ek_floor = total_ek_floor + bio_mixer.ek_floor_cost

                if mamba_out_target is not None:
                    per_block_cosine[i] = chunk_cos_sum / max(accum, 1)

                if accum > 1:
                    bio_out_cat = torch.cat(bio_outs, dim=0)
                else:
                    bio_out_cat = bio_outs[0]
                hidden_states = hidden_states + bio_out_cat
                del h_chunks, bio_outs, bio_out_cat

            else:
                # 非 Mamba 层: 冻结 forward (gradient checkpoint)
                if block.block_type == "attention":
                    layer_mask = causal_mask
                elif block.block_type == "mamba":
                    # 未替换的 Mamba 层 (不应出现, 但防御性处理)
                    layer_mask = mamba_mask
                else:
                    layer_mask = None

                if self.training:
                    _fn = self._make_frozen_fn(block, cache_position, layer_mask)
                    hidden_states = ckpt_fn(_fn, hidden_states, use_reentrant=False)
                else:
                    hidden_states = block(
                        hidden_states, cache_params=None,
                        cache_position=cache_position,
                        attention_mask=layer_mask,
                    )

        # ===== KL + CE 分块计算 =====
        lm_head = self.nvidia_model.lm_head
        lm_dtype = lm_head.weight.dtype
        norm_f = backbone.norm_f

        h_chunks = hidden_states.chunk(accum, dim=0)
        t_chunks = teacher_hidden_final.chunk(accum, dim=0)
        l_chunks = labels.chunk(accum, dim=0)
        m_chunks = loss_mask.chunk(accum, dim=0)

        kl_sum = torch.tensor(0.0, device=input_ids.device)
        ce_sum = torch.tensor(0.0, device=input_ids.device)
        mask_sum = torch.tensor(0.0, device=input_ids.device)

        # KL 温度从外部通过 _kl_temperature 传入
        T = getattr(self, '_kl_temperature', 4.0)

        for h_c, t_c, l_c, m_c in zip(h_chunks, t_chunks, l_chunks, m_chunks):
            # Gradient checkpoint: 避免全量 logits 留在 autograd graph
            _fn = self._make_kl_ce_fn(norm_f, lm_head, lm_dtype, m_c, T)
            ce_val, kl_val, mask_val = ckpt_fn(
                _fn, h_c, t_c, l_c, use_reentrant=False)
            ce_sum = ce_sum + ce_val
            kl_sum = kl_sum + kl_val
            mask_sum = mask_sum + mask_val

        mask_total = mask_sum.clamp(min=1)
        ce_loss = ce_sum / mask_total
        kl_loss = kl_sum / mask_total

        # MSE 平均
        mse_loss = total_mse / max(mse_count, 1)

        # PonderNet 平均
        n_ponder = max(self._num_mamba * accum, 1)
        ponder_cost = total_ponder / n_ponder
        ek_floor_cost = total_ek_floor / n_ponder

        return Stage2Output(
            kl_loss=kl_loss,
            ce_loss=ce_loss,
            mse_loss=mse_loss,
            ponder_cost=ponder_cost,
            ek_floor_cost=ek_floor_cost,
            per_block_cosine=per_block_cosine,
        )

    # ====== 复用 distill_wrapper_v3 的梯度操作 ======

    def compensate_modulation_gradients(self, max_comp: float = 100.0):
        """Natural Gradient 补偿: sigmoid/softplus 饱和补偿。"""
        for bio_mixer in self.bio_ssm_modules.values():
            if not any(p.requires_grad for p in bio_mixer.parameters()):
                continue
            snn_block = bio_mixer.bio_ssm.snn_block

            if snn_block.b_beta.grad is not None:
                with torch.no_grad():
                    beta = torch.sigmoid(snn_block.b_beta.data)
                    sigmoid_deriv = (beta * (1.0 - beta)).clamp(min=1.0 / max_comp)
                    snn_block.b_beta.grad.div_(sigmoid_deriv)

            if snn_block.b_alpha.grad is not None:
                with torch.no_grad():
                    softplus_deriv = torch.sigmoid(snn_block.b_alpha.data).clamp(min=0.1)
                    snn_block.b_alpha.grad.div_(softplus_deriv)

    def clip_halt_proj_gradients(self, max_norm: float = 0.5):
        """halt_proj 独立梯度裁剪。"""
        halt_params = []
        for bio_mixer in self.bio_ssm_modules.values():
            for p in bio_mixer.bio_ssm.halt_proj.parameters():
                if p.grad is not None:
                    halt_params.append(p)
        if halt_params:
            torch.nn.utils.clip_grad_norm_(halt_params, max_norm)

    def get_active_param_groups(self, lr=2e-4, neuron_lr_mult=10.0, weight_decay=0.1):
        """只收集 requires_grad=True 的参数, 分 decay/no_decay/neuron 三组。"""
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
        groups = []
        if decay:
            groups.append({'params': decay, 'lr': lr, 'weight_decay': weight_decay, 'lr_mult': 1.0})
        if no_decay:
            groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0, 'lr_mult': 1.0})
        if neuron:
            groups.append({'params': neuron, 'lr': lr * neuron_lr_mult, 'weight_decay': 0.0,
                           'lr_mult': float(neuron_lr_mult)})
        return groups

    # ====== Checkpoint ======

    def save_bio_ssm_state(self):
        """提取仅 BioSSM 模块的 state_dict。"""
        return {k: v.cpu() for k, v in self.bio_ssm_modules.state_dict().items()}

    def load_bio_ssm_state(self, state_dict):
        """加载 BioSSM 模块的 state_dict。"""
        self.bio_ssm_modules.load_state_dict(state_dict)
