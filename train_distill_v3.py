"""
v3 蒸馏训练: Nemotron-3-Nano Mamba → BioSSM (FSDP 多卡)

核心设计:
  1. 加载 NemotronHForCausalLM (pretrained), 冻结全部原始参数
  2. 在每个 Mamba 位置并行挂载 BioSSMLayer (可训练)
  3. Loss: CE + per-layer cosine alignment + ponder + ek_floor
  4. 只有 BioSSMLayer 参数有梯度和 optimizer states
  5. FSDP 分片全模型, use_orig_params=True
  6. Checkpoint 只保存 BioSSM 权重 (不保存 30B 冻结模型)

显存 (4×48GB, FSDP, N=4):
  模型分片:     ~15GB/卡 (30B bf16 / 4)
  BioSSM:       ~1GB/卡
  Optimizer:    ~6GB/卡 (仅 BioSSM)
  Activations:  ~24GB/卡 剩余

加载策略 (防 OOM):
  - 所有 rank 同种子独立加载到 CPU (low_cpu_mem_usage=True)
  - FSDP 不用 sync_module_states (避免全量移到 rank0 GPU)
  - FSDP 逐层分片, 每张卡只放 1/N 参数

用法:
  torchrun --nproc_per_node=4 train_distill_v3.py \
      --teacher_path /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --data_path data/xxx.jsonl \
      --N 4 --K 16 --batch_size 1 --accumulation_steps 16
"""

import os
import sys
import time
import math
import glob
import argparse
import warnings
import functools
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

from distill_wrapper_v3 import DistillHybridModel, BioSSMConfig
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
# WSD 学习率调度
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
# 蒸馏调度 (从 train_distill.py 移植)
# ============================================================

def get_distill_schedule(step, total_steps):
    """3 阶段蒸馏权重。

    Returns: (alpha_ce, beta_hidden)
      Phase 1 (0-30%):   强对齐 — α=0.3, β=2.0
      Phase 2 (30-70%):  过渡   — α→1.0, β→0.3
      Phase 3 (70-100%): 自主   — α=1.0, β=0.1
    """
    frac = step / max(total_steps, 1)
    if frac < 0.3:
        return 0.3, 2.0
    elif frac < 0.7:
        t = (frac - 0.3) / 0.4
        return 0.3 + 0.7 * t, 2.0 - 1.7 * t
    else:
        return 1.0, 0.1


# ============================================================
# 序列长度课程 (从 train_distill.py 移植)
# ============================================================

def get_curriculum_seq_len(step, total_steps, max_length):
    """渐进式序列长度: 64 → 128 → 256 → max_length。"""
    frac = step / max(total_steps, 1)
    if frac < 0.15:
        return min(64, max_length)
    elif frac < 0.35:
        return min(128, max_length)
    elif frac < 0.60:
        return min(256, max_length)
    return max_length


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, step, args, rank):
    """只保存 BioSSM 权重 + optimizer (不含冻结的 30B 模型)。"""
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'distill_v3_step{step}.pth')

    # 从 FSDP 中提取 BioSSM state_dict
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        full_sd = model.state_dict()

    if rank == 0:
        # 只保存 bio_ssm_modules 的参数 (剥离前缀, 与 load_bio_ssm_state 对齐)
        prefix = 'bio_ssm_modules.'
        bio_sd = {k[len(prefix):]: v for k, v in full_sd.items()
                  if k.startswith(prefix)}
        torch.save({
            'bio_ssm': bio_sd,
            'step': step,
            'config': {
                'hidden_size': args.hidden_size,
                'ssm_N': args.N,
                'ssm_K': args.K,
                'ssm_v_th_min': args.v_th_min,
                'ssm_ek_floor': args.ek_floor,
                'num_hidden_layers': args.num_layers,
                'teacher_path': args.teacher_path,
            },
        }, path)
        Log(f'  → BioSSM checkpoint: {path} ({len(bio_sd)} keys)')

    dist.barrier()

    # 清理旧 checkpoint (保留最近 3 个)
    if rank == 0:
        ckpts = sorted(glob.glob(os.path.join(args.out_dir, 'distill_v3_step*.pth')))
        for old in ckpts[:-3]:
            os.remove(old)


# ============================================================
# 训练循环
# ============================================================

def train_epoch(model, raw_model, train_loader, sampler, optimizer, ctx,
                dashboard, args, epoch, iter_per_epoch, tokens_seen,
                rank, world_size):
    """单 epoch 蒸馏训练。"""
    sampler.set_epoch(epoch)
    model.train()
    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])
    total_iters = args.epochs * iter_per_epoch

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        global_step = epoch * iter_per_epoch + step

        # 序列长度课程
        if not args.no_curriculum:
            curr_seq = get_curriculum_seq_len(global_step, total_iters, args.max_length)
            X = X[:, :curr_seq]
            Y = Y[:, :curr_seq]
            loss_mask = loss_mask[:, :curr_seq]

        X = X.to(f"cuda:{local_rank}", non_blocking=True)
        Y = Y.to(f"cuda:{local_rank}", non_blocking=True)
        loss_mask = loss_mask.to(f"cuda:{local_rank}", non_blocking=True)

        # 学习率
        opt_step = global_step // args.accumulation_steps
        lr = get_lr_wsd(opt_step, total_iters // args.accumulation_steps,
                        args.learning_rate, args.warmup_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_mult', 1.0)

        # 蒸馏调度 (固定值优先)
        if args.alpha_ce is not None and args.beta_hidden is not None:
            alpha_ce, beta_hidden = args.alpha_ce, args.beta_hidden
        else:
            alpha_ce, beta_hidden = get_distill_schedule(global_step, total_iters)

        is_boundary = (step + 1) % args.accumulation_steps == 0
        sync_ctx = nullcontext() if is_boundary else model.no_sync()

        with sync_ctx:
            with ctx:
                out = model(X, Y, loss_mask=loss_mask)

                loss = torch.tensor(0.0, device=X.device)

                # CE loss
                if out.ce_loss is not None:
                    loss = loss + alpha_ce * out.ce_loss

                # Hidden alignment (cosine)
                if out.hidden_loss is not None:
                    loss = loss + beta_hidden * out.hidden_loss

                # SNN 正则
                if out.ponder_cost is not None and args.ponder_weight > 0:
                    loss = loss + args.ponder_weight * out.ponder_cost
                if out.ek_floor_cost is not None and args.ek_floor_weight > 0:
                    loss = loss + args.ek_floor_weight * out.ek_floor_cost

                loss = loss / args.accumulation_steps

            loss.backward()

        if is_boundary:
            # Dashboard: 缓存梯度范数 (all ranks 参与 all_reduce)
            if step % args.log_interval < args.accumulation_steps:
                dashboard.cache_grad_norms(model)

            raw_model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Token 统计
        valid_tokens = loss_mask.view(-1).sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        # 日志
        if is_boundary and step % args.log_interval < args.accumulation_steps:
            batch_loss = loss.item() * args.accumulation_steps
            elapsed = time.time() - start_time
            tps = tokens_seen / max(elapsed, 1)
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            seq = X.shape[1]

            ce_val = out.ce_loss.item() if out.ce_loss is not None else None
            hid_val = out.hidden_loss.item() if out.hidden_loss is not None else None
            pc_val = out.ponder_cost.item() if torch.is_tensor(out.ponder_cost) else None
            ek_val = out.ek_floor_cost.item() if torch.is_tensor(out.ek_floor_cost) else None

            # Dashboard TensorBoard
            dashboard.log_step(opt_step, {
                'loss': batch_loss,
                'ce_loss': ce_val,
                'cosine_loss': hid_val,
                'ponder_cost': pc_val,
                'ek_floor_cost': ek_val,
                'alpha_ce': alpha_ce,
                'beta_hidden': beta_hidden,
                'lr': lr,
                'tps': tps,
                'tokens_seen': tokens_seen,
                'seq_len': seq,
                'memory_current_gb': mem_cur,
                'memory_peak_gb': mem_peak,
            }, raw_model)

            # 终端输出
            if rank == 0:
                ce_str = f'{ce_val:.3f}' if ce_val is not None else '-'
                hid_str = f'{hid_val:.4f}' if hid_val is not None else '-'
                pc_str = f'{pc_val:.2f}' if pc_val is not None else '-'

                print(
                    f'E{epoch}[{step}/{iter_per_epoch}] '
                    f'loss:{batch_loss:.3f} CE:{ce_str} cos:{hid_str} E[K]:{pc_str} '
                    f'α={alpha_ce:.2f} β={beta_hidden:.2f} lr:{lr:.2e} seq:{seq} '
                    f'TPS:{tps:.0f} Mem:{mem_cur:.1f}/{mem_peak:.1f}GB '
                    f'GPUs:{world_size}',
                    flush=True,
                )

        # 保存
        if is_boundary and opt_step > 0 and opt_step % args.save_interval == 0:
            dashboard.log_save_point(opt_step, raw_model)
            model.eval()
            save_checkpoint(model, optimizer, global_step, args, rank)
            model.train()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='NeuronSpark v3 Mamba→BioSSM 蒸馏 (FSDP)')

    # Teacher
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='NVIDIA Nemotron-3 模型目录 (含 modeling/config py)')

    # BioSSM 参数
    parser.add_argument('--N', type=int, default=4, help='SSM 状态扩展因子')
    parser.add_argument('--K', type=int, default=16, help='最大 SNN 时间步')
    parser.add_argument('--v_th_min', type=float, default=0.1)
    parser.add_argument('--ek_floor', type=float, default=4.0)

    # 训练
    parser.add_argument('--out_dir', type=str, default='checkpoints_distill_v3')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1, help='per-GPU')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)

    # 优化
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--accumulation_steps', type=int, default=16)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # SNN 正则
    parser.add_argument('--ponder_weight', type=float, default=0.01)
    parser.add_argument('--ek_floor_weight', type=float, default=0.1)

    # 蒸馏
    parser.add_argument('--no_curriculum', action='store_true',
                        help='关闭序列长度课程学习')
    parser.add_argument('--alpha_ce', type=float, default=None,
                        help='固定 CE 权重 (覆盖自动调度)')
    parser.add_argument('--beta_hidden', type=float, default=None,
                        help='固定 cosine alignment 权重 (覆盖自动调度)')

    # FSDP
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=['full_shard', 'shard_grad_op', 'no_shard'])

    # 日志 & Checkpoint
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard 日志目录 (None=禁用)')

    # 数据
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据路径: JSONL 文件 / HF 数据集目录或名称 / Parquet 文件')
    parser.add_argument('--data_type', type=str, default='jsonl',
                        choices=['jsonl', 'huggingface', 'parquet'],
                        help='数据格式 (默认 jsonl)')
    parser.add_argument('--text_column', type=str, default='text',
                        help='文本列名 (默认 text)')
    parser.add_argument('--data_split', type=str, default='train',
                        help='HF 数据集 split (默认 train)')
    parser.add_argument('--data_subset', type=str, default=None,
                        help='HF 数据集子集名称')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='tokenizer 路径 (默认用 teacher_path)')

    # 恢复
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复 BioSSM checkpoint 路径')

    args = parser.parse_args()

    # ==================== 分布式 ====================
    local_rank, rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"

    # 统一种子: 保证所有 rank 的 BioSSM 初始化相同 (避免 sync_module_states)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # ==================== 加载 Teacher ====================
    Log(f'加载 Teacher: {args.teacher_path}', rank)
    sys.path.insert(0, args.teacher_path)

    from transformers import AutoModelForCausalLM, AutoConfig
    nvidia_config = AutoConfig.from_pretrained(args.teacher_path, trust_remote_code=True)

    # 所有 rank 独立加载到 CPU (同种子 → 同权重, 无需 sync_module_states)
    # low_cpu_mem_usage: 逐层加载, 减少 CPU 峰值内存
    nvidia_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # 从 teacher config 读取维度
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

    # BioSSM 初始化用统一种子 (torch.manual_seed(42) 已在上方设置)
    distill_model = DistillHybridModel(nvidia_model, bio_config)

    n_mamba = distill_model._num_mamba
    bio_params = sum(p.numel() for p in distill_model.bio_ssm_modules.parameters())
    Log(f'  BioSSM: {n_mamba} 层, {bio_params/1e6:.1f}M 可训练参数 '
        f'(N={args.N}, K={args.K})', rank)

    # 恢复 BioSSM checkpoint
    if args.resume:
        Log(f'恢复 BioSSM: {args.resume}', rank)
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        distill_model.load_bio_ssm_state(ckpt['bio_ssm'])
        del ckpt

    # 参数分组 (FSDP wrap 之前)
    param_groups = distill_model.get_bio_ssm_param_groups(
        lr=args.learning_rate,
        neuron_lr_mult=args.neuron_lr_mult,
        weight_decay=args.weight_decay,
    )

    # 自动 dtype 对齐: FSDP flatten 要求同 wrap 单元内 dtype 一致
    # 扫描所有参数和 buffer, 统计 dtype 分布, 对齐到多数 dtype
    from collections import Counter
    dtype_counts = Counter()
    dtype_names = {}  # dtype → 示例参数名 (用于日志)
    for name, p in distill_model.named_parameters():
        dtype_counts[p.dtype] += 1
        if p.dtype not in dtype_names:
            dtype_names[p.dtype] = name
    for name, buf in distill_model.named_buffers():
        dtype_counts[buf.dtype] += 1
        if buf.dtype not in dtype_names:
            dtype_names[buf.dtype] = name

    if len(dtype_counts) > 1:
        target_dtype = dtype_counts.most_common(1)[0][0]
        Log(f'  dtype 分布: {dict(dtype_counts)}', rank)
        for dt, example in dtype_names.items():
            if dt != target_dtype:
                Log(f'    {dt} → {target_dtype} (例: {example})', rank)
        distill_model.to(target_dtype)
        Log(f'  全模型 dtype 对齐: {target_dtype}', rank)
    else:
        target_dtype = dtype_counts.most_common(1)[0][0]
        Log(f'  dtype 统一: {target_dtype}', rank)

    raw_model = distill_model

    # ==================== FSDP ====================
    # 从已加载的模型实例获取实际的 block 类
    # 注意: trust_remote_code=True 加载的类在 transformers_modules 命名空间下,
    # 与 sys.path 直接导入的 NemotronHBlock 是不同的类对象!
    # isinstance() 按类对象身份匹配, 必须用模型实例中的实际类
    ActualBlockCls = type(nvidia_model.backbone.layers[0])
    Log(f'  FSDP wrap 类: {ActualBlockCls.__module__}.{ActualBlockCls.__name__}', rank)

    # 验证: 类匹配检查
    match_count = sum(1 for layer in nvidia_model.backbone.layers
                      if isinstance(layer, ActualBlockCls))
    Log(f'  类匹配验证: {match_count}/{len(nvidia_model.backbone.layers)} 层命中 '
        f'(应为 {len(nvidia_model.backbone.layers)})', rank)

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ActualBlockCls, BioSSMLayer},
    )

    sharding_map = {
        'full_shard': ShardingStrategy.FULL_SHARD,
        'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
        'no_shard': ShardingStrategy.NO_SHARD,
    }

    # 关键: 不用 sync_module_states (避免 FSDP 把完整 60GB 模型移到 rank0 GPU)
    # 所有 rank 已用相同种子加载相同权重, 无需广播
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

    # FSDP 分片后显存验证
    fsdp_gpu_mb = torch.cuda.memory_allocated(local_rank) / 1e6
    fsdp_gpu_gb = fsdp_gpu_mb / 1000
    Log(f'  FSDP 分片后 GPU{local_rank} 显存: {fsdp_gpu_gb:.2f} GB '
        f'(预期 ~17GB, 若 >40GB 则分片失败)', rank)

    # 统计 FSDP 内部 wrap 数量
    fsdp_count = sum(1 for m in model.modules()
                     if isinstance(m, FSDP) and m is not model)
    Log(f'  FSDP 内部 wrap 数: {fsdp_count} (预期 ~{len(nvidia_model.backbone.layers) + n_mamba})',
        rank)

    # FSDP 分片完成后, 恢复 per-rank 随机种子 (数据 shuffle 需要不同种子)
    torch.manual_seed(42 + rank)

    # ==================== Tokenizer ====================
    tok_path = args.tokenizer_path or args.teacher_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    Log(f'  Tokenizer: vocab={tokenizer.vocab_size}', rank)

    # ==================== 数据 ====================
    Log(f'  数据: {args.data_path} (type={args.data_type}, col={args.text_column})', rank)
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

    # ==================== Optimizer ====================
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.95))

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps * world_size

    Log(f"\n{'='*60}", rank)
    Log(f"NeuronSpark v3 蒸馏: Mamba → BioSSM (FSDP, {world_size} GPUs)", rank)
    Log(f"  Teacher:      {args.teacher_path}", rank)
    Log(f"  BioSSM:       N={args.N}, K={args.K}, {bio_params/1e6:.1f}M params", rank)
    Log(f"  Mamba 层数:    {n_mamba}", rank)
    Log(f"  Data:         {args.data_path}", rank)
    Log(f"  Batch:        {args.batch_size}/gpu × {world_size} × accum "
        f"{args.accumulation_steps} = {effective_batch}", rank)
    Log(f"  LR:           {args.learning_rate} (WSD, warmup {args.warmup_iters})", rank)
    Log(f"  Neuron LR:    {args.learning_rate * args.neuron_lr_mult} "
        f"({args.neuron_lr_mult}×)", rank)
    Log(f"  Loss:         CE + cosine_align + ponder({args.ponder_weight}) "
        f"+ ek_floor({args.ek_floor_weight})", rank)
    if args.alpha_ce is not None:
        Log(f"  α/β 固定:     α={args.alpha_ce}, β={args.beta_hidden}", rank)
    else:
        Log(f"  α/β 调度:     3阶段自动 (0.3/2.0 → 1.0/0.1)", rank)
    Log(f"  Curriculum:   {'OFF' if args.no_curriculum else '64→128→256→max'}", rank)
    Log(f"  TensorBoard:  {args.log_dir or 'OFF'}", rank)
    Log(f"{'='*60}\n", rank)

    # ==================== Dashboard ====================
    dashboard = DistillDashboard(log_dir=args.log_dir, model=raw_model, rank=rank)

    # ==================== 训练 ====================
    tokens_seen = 0
    for epoch in range(args.epochs):
        tokens_seen = train_epoch(
            model, raw_model, train_loader, sampler, optimizer, ctx,
            dashboard, args, epoch, iter_per_epoch, tokens_seen,
            rank, world_size,
        )

    # ==================== 最终保存 ====================
    save_checkpoint(model, optimizer, args.epochs * iter_per_epoch, args, rank)
    dashboard.close()
    Log(f"\n蒸馏完成! Tokens seen: {tokens_seen:,}", rank)
    Log(f"Peak CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
