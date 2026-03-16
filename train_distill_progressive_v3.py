"""
v3 渐进蒸馏训练: Mamba → BioSSM (FSDP, 两阶段)

Stage 1: 逐块对齐 (back-to-front, 23 个阶段)
  - 纯 teacher forward → 捕获 (block_input, mamba_output)
  - 单块 BioSSM 训练: MSE + cosine + ponder + ek_floor
  - 每块 ~1000 步, cos_sim > threshold 提前结束
  - 内存: ~23GB/卡

Stage 2: 端到端整合 (渐进解冻)
  - 加载 Stage 1 权重 → 渐进解冻 [1→2→4→8→16→23]
  - Dual forward: teacher(no_grad) + student(BioSSM)
  - Loss: KL + CE + MSE + ponder
  - 温度 T: 4.0→1.0 线性衰减
  - 内存: 最终阶段 ~40GB/卡

用法:
  # Stage 1
  torchrun --nproc_per_node=4 train_distill_progressive_v3.py \
      --teacher_path /path/to/NVIDIA-Nemotron --data_path data/xxx.jsonl \
      --stage 1 --stage1_steps 1000 --stage1_lr 5e-4

  # Stage 2 (加载 Stage 1 权重)
  torchrun --nproc_per_node=4 train_distill_progressive_v3.py \
      --teacher_path /path/to/NVIDIA-Nemotron --data_path data/xxx.jsonl \
      --stage 2 --resume checkpoints_distill_v3/stage1/stage1_complete.pth

  # Both stages
  torchrun --nproc_per_node=4 train_distill_progressive_v3.py \
      --teacher_path /path/to/NVIDIA-Nemotron --data_path data/xxx.jsonl \
      --stage both
"""

import os
import sys
import time
import math
import glob
import argparse
import warnings
import functools
from collections import Counter
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import AutoTokenizer

from distill_progressive_v3 import ProgressiveDistillModel, BioSSMConfig
from atomic_ops.bio_ssm_layer import BioSSMLayer
from dashboard_distill import DistillDashboard
from dataset_distill import create_distill_dataset

warnings.filterwarnings('ignore')


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def Log(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)


# ============================================================
# WSD 学习率调度 (复用 train_distill_v3.py)
# ============================================================

def get_lr_wsd(step, total_steps, lr, warmup_steps, stable_ratio=0.8):
    """Warmup → Stable → sqrt Decay。"""
    min_lr = lr / 100
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    stable_end = warmup_steps + int((total_steps - warmup_steps) * stable_ratio)
    if step <= stable_end:
        return lr
    if step > total_steps:
        return min_lr
    decay_progress = (step - stable_end) / max(total_steps - stable_end, 1)
    return min_lr + (1.0 - math.sqrt(decay_progress)) * (lr - min_lr)


# ============================================================
# 序列长度课程
# ============================================================

def get_curriculum_seq_len(step, total_steps, max_length):
    """渐进式序列长度: 64 → 128 → 256 → max_length (加速版)。"""
    frac = step / max(total_steps, 1)
    if frac < 0.05:
        return min(64, max_length)
    elif frac < 0.15:
        return min(128, max_length)
    elif frac < 0.30:
        return min(256, max_length)
    return max_length


# ============================================================
# Checkpoint 工具
# ============================================================

def save_bio_ssm_checkpoint(model, step, save_dir, tag, args, rank):
    """保存 BioSSM 权重 (从 FSDP 提取)。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{tag}_step{step}.pth')

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        full_sd = model.state_dict()

    if rank == 0:
        prefix = 'bio_ssm_modules.'
        bio_sd = {k[len(prefix):]: v for k, v in full_sd.items()
                  if k.startswith(prefix)}
        torch.save({
            'bio_ssm': bio_sd,
            'step': step,
            'tag': tag,
            'config': {
                'hidden_size': args.hidden_size,
                'ssm_N': args.N,
                'ssm_K': args.K,
                'ssm_v_th_min': args.v_th_min,
                'ssm_ek_floor': args.ek_floor,
                'num_hidden_layers': args.num_layers,
            },
        }, path)
        Log(f'  → Checkpoint: {path} ({len(bio_sd)} keys)')

    dist.barrier()
    return path


def cleanup_old_checkpoints(save_dir, pattern, keep=3, rank=0):
    """清理旧 checkpoint, 保留最近 keep 个。"""
    if rank != 0:
        return
    ckpts = sorted(glob.glob(os.path.join(save_dir, pattern)))
    for old in ckpts[:-keep]:
        os.remove(old)


# ============================================================
# Stage 1: 逐块训练
# ============================================================

def stage1_train_all(fsdp_model, raw_model, train_loader, sampler, ctx,
                     dashboard, args, rank, world_size):
    """逐块训练: reversed(mamba_indices), 每块 ~stage1_steps 步。"""
    Log(f"\n{'='*60}", rank)
    Log(f"Stage 1: 逐块对齐 (back-to-front, {raw_model._num_mamba} 块)", rank)
    Log(f"  每块步数: {args.stage1_steps}, LR: {args.stage1_lr}", rank)
    Log(f"  cos 提前结束阈值: {args.stage1_cos_threshold}", rank)
    Log(f"{'='*60}\n", rank)

    stage1_dir = os.path.join(args.out_dir, 'stage1')
    accum = args.accumulation_steps
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"

    for block_order, mamba_idx in enumerate(reversed(raw_model.mamba_indices)):
        Log(f"\n--- Stage 1 Block {block_order+1}/{raw_model._num_mamba}: "
            f"Mamba 层 {mamba_idx} ---", rank)

        # 设置当前训练块
        raw_model.set_stage1_block(mamba_idx)

        # 重建 optimizer (只含活跃参数)
        param_groups = raw_model.get_active_param_groups(
            lr=args.stage1_lr,
            neuron_lr_mult=args.neuron_lr_mult,
            weight_decay=args.weight_decay,
        )
        if not param_groups:
            Log(f"  [跳过] 无可训练参数", rank)
            continue
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.95))

        # 训练循环
        fsdp_model.train()
        sampler.set_epoch(block_order)
        micro_X = []
        best_cos = 1.0
        opt_step = 0
        tokens_seen = 0
        start_time = time.time()

        for step, (X, Y, loss_mask) in enumerate(train_loader):
            if opt_step >= args.stage1_steps:
                break

            # 序列长度课程
            if not args.no_curriculum:
                curr_seq = get_curriculum_seq_len(
                    opt_step, args.stage1_steps, args.max_length)
                X = X[:, :curr_seq]
                Y = Y[:, :curr_seq]
                loss_mask = loss_mask[:, :curr_seq]

            micro_X.append(X.to(device, non_blocking=True))

            is_boundary = (step + 1) % accum == 0
            if not is_boundary:
                continue

            # 拼接 micro-batches
            X_cat = torch.cat(micro_X, dim=0)
            n_micro = len(micro_X)
            micro_X = []

            # 学习率
            lr = get_lr_wsd(opt_step, args.stage1_steps,
                            args.stage1_lr, args.warmup_iters)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * pg.get('lr_mult', 1.0)

            # Forward (通过 FSDP wrapper, 触发顶层 all-gather)
            with ctx:
                out = fsdp_model(X_cat, accumulation_steps=n_micro)

                loss = out.mse_loss + 0.5 * out.cos_loss
                if args.ponder_weight > 0:
                    loss = loss + args.ponder_weight * out.ponder_cost
                if args.ek_floor_weight > 0:
                    loss = loss + args.ek_floor_weight * out.ek_floor_cost

            loss.backward()

            # Gradient ops
            raw_model.compensate_modulation_gradients()
            raw_model.clip_halt_proj_gradients(args.halt_grad_clip)
            torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Token 统计
            valid_tokens = X_cat.numel()
            tokens_seen += valid_tokens * world_size
            opt_step += 1

            # Cosine similarity (1 - cos_loss)
            cos_sim = 1.0 - out.cos_loss.item()
            best_cos = min(best_cos, out.cos_loss.item())

            # 日志
            if opt_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                tps = tokens_seen / max(elapsed, 1)
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9

                dashboard.log_stage1_step(opt_step, mamba_idx, {
                    'mse_loss': out.mse_loss.item(),
                    'cos_loss': out.cos_loss.item(),
                    'cos_sim': cos_sim,
                    'ponder_cost': out.ponder_cost.item(),
                    'ek_floor_cost': out.ek_floor_cost.item(),
                    'loss': loss.item(),
                    'lr': lr,
                    'tps': tps,
                    'memory_current_gb': mem_cur,
                    'memory_peak_gb': mem_peak,
                })

                if rank == 0:
                    print(
                        f'  S1[{mamba_idx}] step {opt_step}/{args.stage1_steps} '
                        f'MSE:{out.mse_loss.item():.4f} '
                        f'cos:{cos_sim:.4f} '
                        f'E[K]:{out.ponder_cost.item():.2f} '
                        f'lr:{lr:.2e} TPS:{tps:.0f} '
                        f'Mem:{mem_cur:.1f}/{mem_peak:.1f}GB',
                        flush=True,
                    )

            del X_cat, out

            # 提前结束
            if cos_sim > args.stage1_cos_threshold:
                Log(f'  cos_sim={cos_sim:.4f} > {args.stage1_cos_threshold}, '
                    f'提前结束 (step {opt_step})', rank)
                break

        # 保存当前块
        save_bio_ssm_checkpoint(
            fsdp_model, opt_step, stage1_dir,
            f'block_{mamba_idx}', args, rank,
        )
        Log(f'  Block {mamba_idx} 完成: {opt_step} steps, '
            f'best_cos_loss={best_cos:.4f}', rank)

    # 保存 Stage 1 完整权重
    save_bio_ssm_checkpoint(
        fsdp_model, 0, stage1_dir, 'stage1_complete', args, rank,
    )
    Log(f"\nStage 1 完成!", rank)


# ============================================================
# Stage 2: 端到端整合
# ============================================================

def stage2_progressive_train(fsdp_model, raw_model, train_loader, sampler, ctx,
                             dashboard, args, rank, world_size):
    """渐进端到端: phases 定义活跃块数递增序列。"""
    phases = [int(x) for x in args.stage2_phases.split(',')]
    reversed_mamba = list(reversed(raw_model.mamba_indices))

    Log(f"\n{'='*60}", rank)
    Log(f"Stage 2: 端到端整合 (渐进解冻)", rank)
    Log(f"  Phases: {phases}", rank)
    Log(f"  每阶段步数: {args.stage2_steps_per_phase}, LR: {args.stage2_lr}", rank)
    Log(f"  KL 温度: {args.kl_temperature}", rank)
    Log(f"{'='*60}\n", rank)

    stage2_dir = os.path.join(args.out_dir, 'stage2')
    accum = args.accumulation_steps
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"

    global_opt_step = 0  # 跨 phase 全局 optimizer step

    for phase_idx, n_active in enumerate(phases):
        n_active = min(n_active, len(reversed_mamba))
        active_indices = reversed_mamba[:n_active]

        Log(f"\n--- Stage 2 Phase {phase_idx}: {n_active} 活跃块 ---", rank)
        dashboard.log_phase_transition(global_opt_step, phase_idx, active_indices)

        # 设置活跃块
        raw_model.set_stage2_active_blocks(active_indices)

        # 重建 optimizer
        param_groups = raw_model.get_active_param_groups(
            lr=args.stage2_lr,
            neuron_lr_mult=args.neuron_lr_mult,
            weight_decay=args.weight_decay,
        )
        if not param_groups:
            Log(f"  [跳过] 无可训练参数", rank)
            continue
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.95))

        # 温度衰减: 跨所有 phase 线性
        total_phase_steps = sum(args.stage2_steps_per_phase for _ in phases)

        # 训练循环
        fsdp_model.train()
        sampler.set_epoch(100 + phase_idx)
        micro_X, micro_Y, micro_M = [], [], []
        opt_step = 0
        tokens_seen = 0
        start_time = time.time()

        for step, (X, Y, loss_mask) in enumerate(train_loader):
            if opt_step >= args.stage2_steps_per_phase:
                break

            # 序列长度课程
            if not args.no_curriculum:
                curr_seq = get_curriculum_seq_len(
                    global_opt_step, total_phase_steps, args.max_length)
                X = X[:, :curr_seq]
                Y = Y[:, :curr_seq]
                loss_mask = loss_mask[:, :curr_seq]

            micro_X.append(X.to(device, non_blocking=True))
            micro_Y.append(Y.to(device, non_blocking=True))
            micro_M.append(loss_mask.to(device, non_blocking=True))

            is_boundary = (step + 1) % accum == 0
            if not is_boundary:
                continue

            X_cat = torch.cat(micro_X, dim=0)
            Y_cat = torch.cat(micro_Y, dim=0)
            M_cat = torch.cat(micro_M, dim=0)
            n_micro = len(micro_X)
            micro_X, micro_Y, micro_M = [], [], []

            # 学习率
            lr = get_lr_wsd(opt_step, args.stage2_steps_per_phase,
                            args.stage2_lr, args.warmup_iters)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * pg.get('lr_mult', 1.0)

            # KL 温度衰减
            T_progress = global_opt_step / max(total_phase_steps, 1)
            T = args.kl_temperature * (1.0 - T_progress) + 1.0 * T_progress
            T = max(T, 1.0)
            raw_model._kl_temperature = T

            # CE 权重调度: 0.3 → 1.0
            alpha_ce = args.alpha_ce_start + (args.alpha_ce_end - args.alpha_ce_start) * T_progress

            # Forward (通过 FSDP wrapper, 触发顶层 all-gather)
            with ctx:
                out = fsdp_model(
                    X_cat, Y_cat, loss_mask=M_cat,
                    accumulation_steps=n_micro,
                )

                loss = (args.alpha_kl * out.kl_loss
                        + alpha_ce * out.ce_loss
                        + args.beta_mse * out.mse_loss)
                if args.ponder_weight > 0:
                    loss = loss + args.ponder_weight * out.ponder_cost
                if args.ek_floor_weight > 0:
                    loss = loss + args.ek_floor_weight * out.ek_floor_cost

            loss.backward()

            # Gradient ops
            raw_model.compensate_modulation_gradients()
            raw_model.clip_halt_proj_gradients(args.halt_grad_clip)
            torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Token 统计
            valid_tokens = M_cat.view(-1).sum()
            if world_size > 1:
                dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
            tokens_seen += int(valid_tokens.item())
            opt_step += 1
            global_opt_step += 1

            # 日志
            if opt_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                tps = tokens_seen / max(elapsed, 1)
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9

                metrics = {
                    'kl_loss': out.kl_loss.item(),
                    'ce_loss': out.ce_loss.item(),
                    'mse_loss': out.mse_loss.item(),
                    'ponder_cost': out.ponder_cost.item(),
                    'ek_floor_cost': out.ek_floor_cost.item(),
                    'loss': loss.item(),
                    'alpha_ce': alpha_ce,
                    'alpha_kl': args.alpha_kl,
                    'beta_mse': args.beta_mse,
                    'temperature': T,
                    'lr': lr,
                    'tps': tps,
                    'memory_current_gb': mem_cur,
                    'memory_peak_gb': mem_peak,
                    'per_block_cosine': out.per_block_cosine,
                }
                dashboard.log_stage2_step(global_opt_step, phase_idx, metrics)

                if rank == 0:
                    cos_vals = list(out.per_block_cosine.values())
                    cos_mean = sum(cos_vals) / max(len(cos_vals), 1) if cos_vals else 0.0
                    print(
                        f'  S2[P{phase_idx}/{n_active}blk] step {opt_step} '
                        f'KL:{out.kl_loss.item():.4f} '
                        f'CE:{out.ce_loss.item():.4f} '
                        f'MSE:{out.mse_loss.item():.4f} '
                        f'cos:{cos_mean:.4f} '
                        f'T:{T:.2f} α_ce:{alpha_ce:.2f} '
                        f'lr:{lr:.2e} TPS:{tps:.0f} '
                        f'Mem:{mem_cur:.1f}/{mem_peak:.1f}GB',
                        flush=True,
                    )

            del X_cat, Y_cat, M_cat, out

            # 保存
            if opt_step > 0 and opt_step % args.save_interval == 0:
                save_bio_ssm_checkpoint(
                    fsdp_model, global_opt_step, stage2_dir,
                    f'phase{phase_idx}_{n_active}blocks', args, rank,
                )

        # Phase 结束保存
        save_bio_ssm_checkpoint(
            fsdp_model, global_opt_step, stage2_dir,
            f'phase{phase_idx}_{n_active}blocks', args, rank,
        )
        Log(f'  Phase {phase_idx} 完成: {opt_step} steps', rank)

    # 最终保存
    save_bio_ssm_checkpoint(
        fsdp_model, global_opt_step, stage2_dir, 'final', args, rank,
    )
    Log(f"\nStage 2 完成!", rank)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='NeuronSpark v3 渐进蒸馏: Mamba→BioSSM (FSDP, 两阶段)')

    # Teacher
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='NVIDIA Nemotron-3 模型目录')

    # BioSSM 参数
    parser.add_argument('--N', type=int, default=4, help='SSM 状态扩展因子')
    parser.add_argument('--K', type=int, default=16, help='最大 SNN 时间步')
    parser.add_argument('--v_th_min', type=float, default=0.1)
    parser.add_argument('--ek_floor', type=float, default=4.0)

    # Stage 选择
    parser.add_argument('--stage', type=str, default='both',
                        choices=['1', '2', 'both'],
                        help='选择阶段: 1, 2, both')

    # Stage 1 参数
    parser.add_argument('--stage1_steps', type=int, default=1000,
                        help='每块步数')
    parser.add_argument('--stage1_lr', type=float, default=5e-4,
                        help='Stage 1 学习率 (隔离块可更激进)')
    parser.add_argument('--stage1_cos_threshold', type=float, default=0.85,
                        help='提前结束阈值 (cosine similarity)')

    # Stage 2 参数
    parser.add_argument('--stage2_phases', type=str, default='1,2,4,8,16,23',
                        help='渐进阶段: 活跃块数序列')
    parser.add_argument('--stage2_steps_per_phase', type=int, default=300,
                        help='每阶段步数')
    parser.add_argument('--stage2_lr', type=float, default=2e-4,
                        help='Stage 2 学习率')
    parser.add_argument('--kl_temperature', type=float, default=4.0,
                        help='KL 初始温度 (线性衰减到 1.0)')
    parser.add_argument('--alpha_kl', type=float, default=1.0,
                        help='KL 权重')
    parser.add_argument('--alpha_ce_start', type=float, default=0.3,
                        help='CE 权重起始')
    parser.add_argument('--alpha_ce_end', type=float, default=1.0,
                        help='CE 权重终止')
    parser.add_argument('--beta_mse', type=float, default=1.0,
                        help='MSE 权重')

    # 训练
    parser.add_argument('--out_dir', type=str, default='checkpoints_distill_v3')
    parser.add_argument('--batch_size', type=int, default=1, help='per-GPU')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)

    # 优化
    parser.add_argument('--accumulation_steps', type=int, default=16)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--halt_grad_clip', type=float, default=0.5)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # SNN 正则
    parser.add_argument('--ponder_weight', type=float, default=0.1)
    parser.add_argument('--ek_floor_weight', type=float, default=0.1)

    # 蒸馏
    parser.add_argument('--no_curriculum', action='store_true',
                        help='关闭序列长度课程学习')

    # FSDP
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=['full_shard', 'shard_grad_op', 'no_shard'])

    # 日志 & Checkpoint
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard 日志目录')

    # 数据
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, default='jsonl',
                        choices=['jsonl', 'huggingface', 'parquet'])
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--data_split', type=str, default='train')
    parser.add_argument('--data_subset', type=str, default=None)
    parser.add_argument('--tokenizer_path', type=str, default=None)

    # 恢复
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复 BioSSM checkpoint 路径')

    args = parser.parse_args()

    # ==================== 分布式 ====================
    local_rank, rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"

    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # ==================== 加载 Teacher ====================
    Log(f'加载 Teacher: {args.teacher_path}', rank)
    sys.path.insert(0, args.teacher_path)

    from transformers import AutoModelForCausalLM, AutoConfig
    nvidia_config = AutoConfig.from_pretrained(
        args.teacher_path, trust_remote_code=True)

    nvidia_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    args.hidden_size = nvidia_config.hidden_size
    args.num_layers = nvidia_config.num_hidden_layers

    teacher_params = sum(p.numel() for p in nvidia_model.parameters())
    Log(f'  Teacher: {teacher_params/1e9:.1f}B params, D={args.hidden_size}, '
        f'L={args.num_layers}', rank)

    # ==================== 构建蒸馏模型 ====================
    bio_config = BioSSMConfig(
        hidden_size=args.hidden_size,
        ssm_N=args.N,
        ssm_K=args.K,
        ssm_v_th_min=args.v_th_min,
        ssm_ek_floor=args.ek_floor,
        num_hidden_layers=args.num_layers,
    )

    distill_model = ProgressiveDistillModel(nvidia_model, bio_config)

    n_mamba = distill_model._num_mamba
    bio_params = sum(p.numel() for p in distill_model.bio_ssm_modules.parameters())
    Log(f'  BioSSM: {n_mamba} 层, {bio_params/1e6:.1f}M 可训练参数 '
        f'(N={args.N}, K={args.K})', rank)

    # 恢复 checkpoint
    if args.resume:
        Log(f'恢复 BioSSM: {args.resume}', rank)
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        distill_model.load_bio_ssm_state(ckpt['bio_ssm'])
        del ckpt

    # dtype 对齐
    dtype_counts = Counter()
    for name, p in distill_model.named_parameters():
        dtype_counts[p.dtype] += 1
    for name, buf in distill_model.named_buffers():
        dtype_counts[buf.dtype] += 1
    if len(dtype_counts) > 1:
        target_dtype = dtype_counts.most_common(1)[0][0]
        distill_model.to(target_dtype)
        Log(f'  dtype 对齐: {target_dtype}', rank)

    raw_model = distill_model

    # ==================== FSDP ====================
    ActualBlockCls = type(nvidia_model.backbone.layers[0])
    Log(f'  FSDP wrap 类: {ActualBlockCls.__name__}', rank)

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ActualBlockCls, BioSSMLayer},
    )

    sharding_map = {
        'full_shard': ShardingStrategy.FULL_SHARD,
        'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
        'no_shard': ShardingStrategy.NO_SHARD,
    }

    model = FSDP(
        distill_model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=sharding_map[args.sharding_strategy],
        device_id=torch.device(device),
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=False,
    )

    fsdp_gpu_gb = torch.cuda.memory_allocated(local_rank) / 1e9
    Log(f'  FSDP 分片后 GPU{local_rank}: {fsdp_gpu_gb:.2f} GB', rank)

    torch.manual_seed(42 + rank)

    # ==================== Tokenizer & Data ====================
    tok_path = args.tokenizer_path or args.teacher_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    Log(f'  Tokenizer: vocab={tokenizer.vocab_size}', rank)

    train_ds = create_distill_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        data_type=args.data_type,
        text_column=args.text_column,
        split=args.data_split,
        subset=args.data_subset,
    )
    Log(f'  样本数: {len(train_ds):,}', rank)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        pin_memory=True, drop_last=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # ==================== Dashboard ====================
    dashboard = DistillDashboard(log_dir=args.log_dir, model=raw_model, rank=rank)

    # ==================== 训练信息 ====================
    effective_batch = args.batch_size * args.accumulation_steps * world_size
    Log(f"\n{'='*60}", rank)
    Log(f"NeuronSpark v3 渐进蒸馏 (FSDP, {world_size} GPUs)", rank)
    Log(f"  Stage:        {args.stage}", rank)
    Log(f"  Teacher:      {args.teacher_path}", rank)
    Log(f"  BioSSM:       N={args.N}, K={args.K}, {bio_params/1e6:.1f}M params", rank)
    Log(f"  Mamba 层数:    {n_mamba}", rank)
    Log(f"  Batch:        {args.batch_size}/gpu × {world_size} × accum "
        f"{args.accumulation_steps} = {effective_batch}", rank)
    Log(f"  Curriculum:   {'OFF' if args.no_curriculum else '64→128→256→max'}", rank)
    Log(f"{'='*60}\n", rank)

    # ==================== 执行 ====================
    if args.stage in ('1', 'both'):
        stage1_train_all(
            model, raw_model, train_loader, sampler, ctx,
            dashboard, args, rank, world_size,
        )

    if args.stage in ('2', 'both'):
        stage2_progressive_train(
            model, raw_model, train_loader, sampler, ctx,
            dashboard, args, rank, world_size,
        )

    dashboard.close()
    Log(f"\n渐进蒸馏完成!", rank)
    Log(f"Peak CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
