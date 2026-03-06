"""
分布式 SFT 训练脚本：SNN 语言模型监督微调（FSDP 全分片数据并行）

相比 sft_ddp.py 的核心改进：
  1. FSDP 全分片：参数/梯度/优化器状态按层分片，显存占用 ~1/N
  2. no_sync() 梯度累积：消除 accumulation 期间的冗余通信
  3. bf16 混合精度：参数 bf16，归约 fp32，无需 GradScaler
  4. 通信计算重叠：forward_prefetch + backward_prefetch
  5. Checkpoint 兼容：保存 FULL_STATE_DICT，可用 generate_sample.py 单卡加载

核心差异（vs train_fsdp.py）：
  - 使用 SFTDataset（对话格式 + loss mask 仅计算 assistant 回复）
  - 加载预训练 checkpoint 权重（--pretrained_ckpt）
  - AdamW 优化器（weight_decay=0.01，neuron 参数 decay=0.0）
  - 默认 lr=5e-5, epochs=3, warmup=100

用法：
    # 单机多卡
    torchrun --nproc_per_node=4 sft_fsdp.py \
        --pretrained_ckpt checkpoints/ckpt_step85000.pth \
        --sft_data_path data/sft/sft_data.jsonl \
        --D 1024 --D_ff 3072 --batch_size 4 --accumulation_steps 8

    # 切换分片策略
    torchrun --nproc_per_node=4 sft_fsdp.py \
        --pretrained_ckpt checkpoints/ckpt_step85000.pth \
        --sharding_strategy hybrid_shard

    # 断续训练
    torchrun --nproc_per_node=4 sft_fsdp.py --resume checkpoints_sft/ckpt_step500.pth
"""

import os
import glob
import time
import math
import argparse
import warnings
import functools

import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

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

from model import SNNLanguageModel
from atomic_ops.snn_decoder_layer import SNNDecoderLayer
from dataset import SFTDataset

warnings.filterwarnings('ignore')

# 分片策略名称映射
SHARDING_STRATEGIES = {
    'full_shard': ShardingStrategy.FULL_SHARD,
    'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
    'no_shard': ShardingStrategy.NO_SHARD,
    'hybrid_shard': ShardingStrategy.HYBRID_SHARD,
}


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


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def Logger(content, rank=0):
    if is_main_process(rank):
        print(content)


# ============================================================
# 学习率调度
# ============================================================

def get_lr(it, total_iters, learning_rate, warmup_iters):
    min_lr = learning_rate / 10
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > total_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# Checkpoint（FSDP 专用）
# ============================================================

def save_checkpoint_fsdp(save_dir, model, optimizer, step, epoch, best_loss, tokens_seen,
                         rank, max_keep=5):
    """保存 FSDP checkpoint（FULL_STATE_DICT 模式，兼容单卡加载）。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'ckpt_step{step}.pth')

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()

    full_optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if is_main_process(rank):
        raw_model = model.module
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state': full_optim_state,
            'step': step,
            'epoch': epoch,
            'best_loss': best_loss,
            'tokens_seen': tokens_seen,
            'model_config': {
                'vocab_size': raw_model.vocab_size,
                'D': raw_model.D,
                'N': raw_model.N,
                'K': raw_model.K,
                'num_layers': raw_model.num_layers,
                'D_ff': raw_model.D_ff,
            },
        }, path)
        print(f"  → Checkpoint saved: {path}")

        ckpts = sorted(glob.glob(os.path.join(save_dir, 'ckpt_step*.pth')))
        while len(ckpts) > max_keep:
            old = ckpts.pop(0)
            os.remove(old)
            print(f"  → Removed old checkpoint: {old}")

    dist.barrier()


def load_pretrained_for_fsdp(path, model, rank):
    """加载预训练权重（FSDP 包装前，在 CPU 上加载）。"""
    Logger(f"Loading pretrained weights from {path}...", rank)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'trainable_state_dict' in ckpt:
        model.load_state_dict(ckpt['trainable_state_dict'], strict=False)

    pretrain_step = ckpt.get('step', '?')
    Logger(f"  Loaded pretrained model (step={pretrain_step})", rank)


def load_checkpoint_fsdp(path, model, optimizer, device, rank):
    """加载 SFT checkpoint 恢复训练。"""
    Logger(f"Loading SFT checkpoint from {path}...", rank)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        if 'model_state_dict' in ckpt:
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if missing:
                Logger(f"  New params (random init): {missing}", rank)
            if unexpected:
                Logger(f"  Unexpected keys (ignored): {unexpected}", rank)

    if 'optimizer_state' in ckpt:
        try:
            full_optim_state = ckpt['optimizer_state']
            sharded_optim_state = FSDP.shard_full_optim_state_dict(
                full_optim_state, model
            )
            optimizer.load_state_dict(sharded_optim_state)
        except (ValueError, KeyError, RuntimeError):
            Logger("  Warning: Optimizer state incompatible, starting fresh.", rank)

    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))
    tokens_seen = ckpt.get('tokens_seen', 0)
    Logger(f"  Resumed: step={step}, epoch={epoch}, tokens={tokens_seen:,}", rank)
    return step, epoch, best_loss, tokens_seen


# ============================================================
# 初始化
# ============================================================

def init_model(args, local_rank, rank):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = SNNLanguageModel(
        vocab_size=args.vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'SNN LM 总参数量：{total_params / 1e6:.3f} 百万', rank)

    return model, tokenizer


def wrap_model_fsdp(model, args, local_rank):
    """FSDP 包装模型。"""
    device = torch.device(f"cuda:{local_rank}")

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={SNNDecoderLayer},
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    sharding = SHARDING_STRATEGIES.get(args.sharding_strategy, ShardingStrategy.FULL_SHARD)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=sharding,
        device_id=device,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
    )

    return model, device


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, raw_model, train_loader, sampler, optimizer, ctx, args,
                iter_per_epoch, tokens_seen, rank, world_size, dashboard=None):
    """训练一个 epoch（FSDP SFT 版本）。"""
    sampler.set_epoch(epoch)
    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(f"cuda:{local_rank}", non_blocking=True)
        Y = Y.to(f"cuda:{local_rank}", non_blocking=True)
        loss_mask = loss_mask.to(f"cuda:{local_rank}", non_blocking=True)

        # 学习率调度
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch,
                     args.learning_rate, args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        # no_sync: 梯度累积期间跳过 allreduce
        is_boundary = (step + 1) % args.accumulation_steps == 0
        sync_ctx = nullcontext() if is_boundary else model.no_sync()

        with sync_ctx:
            with ctx:
                out = model(X, Y)
                loss = out.last_loss / args.accumulation_steps
                loss_mask_flat = loss_mask.view(-1)
                loss_mask_float = loss_mask_flat.float()
                mask_sum = loss_mask_float.sum()
                if mask_sum > 0:
                    loss = torch.sum(loss * loss_mask_float) / mask_sum
                else:
                    loss = loss.mean()
                # 动态 K: ponder cost 正则化
                if out.ponder_cost is not None and args.ponder_weight > 0:
                    loss = loss + args.ponder_weight * out.ponder_cost / args.accumulation_steps

            loss.backward()

        if is_boundary:
            raw_model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Dashboard: optimizer.step() 后、zero_grad() 前（梯度仍可用）
            # summon_full_params 是集合操作，所有 rank 必须参与；仅 rank 0 写日志
            need_log = step % args.log_interval == 0
            need_save = (step + 1) % args.save_interval == 0
            if need_log or need_save:
                with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                    if dashboard:
                        global_step = epoch * iter_per_epoch + step
                        if need_log:
                            batch_loss_db = loss.item() * args.accumulation_steps
                            spend_time_db = time.time() - start_time
                            dashboard.log_step(global_step, {
                                'loss': batch_loss_db,
                                'ppl': math.exp(min(batch_loss_db, 20.0)),
                                'lr': lr,
                                'tps': tokens_seen / spend_time_db if spend_time_db > 0 else 0,
                                'tokens_seen': tokens_seen,
                                'ponder_cost': out.ponder_cost.item() if out.ponder_cost is not None else 0.0,
                                'memory_current_gb': torch.cuda.memory_allocated() / 1e9,
                                'memory_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
                            }, raw_model, log_params=True)
                        if need_save:
                            dashboard.log_save_point(global_step, raw_model)

            optimizer.zero_grad(set_to_none=True)

        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        if step % args.log_interval == 0 and is_main_process(rank):
            batch_loss = loss.item() * args.accumulation_steps
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                'SFT Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{} [FSDP]'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, batch_ppl, lr, tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

        if is_boundary and (step + 1) % args.save_interval == 0:
            model.eval()
            global_step = epoch * iter_per_epoch + step + 1
            cur_loss = loss.item() * args.accumulation_steps
            save_checkpoint_fsdp(args.save_dir, model, optimizer,
                                 global_step, epoch, cur_loss, tokens_seen, rank)
            model.train()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model SFT (FSDP)")

    # SNN 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)

    # SFT 特有
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--sft_data_path', type=str, default='data/sft/sft_data.jsonl')

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="DataLoader prefetch_factor")

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)
    parser.add_argument('--ponder_weight', type=float, default=0.01,
                        help='动态 K ponder cost 正则化权重')

    # FSDP 参数
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=list(SHARDING_STRATEGIES.keys()),
                        help="FSDP 分片策略")

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 数据
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    # TensorBoard 看板（不传则禁用，零开销）
    parser.add_argument('--dashboard_dir', type=str, default=None,
                        help="TensorBoard 日志目录，例如 runs/sft")

    args = parser.parse_args()

    # ==================== 分布式初始化 ====================
    local_rank, rank, world_size = setup_distributed()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42 + rank)

    # 性能优化
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # ==================== 模型初始化 ====================
    model, tokenizer = init_model(args, local_rank, rank)

    # 加载预训练权重（FSDP 包装前，仅 rank 0 需要加载，sync_module_states 会广播）
    if args.pretrained_ckpt and not args.resume:
        if is_main_process(rank):
            load_pretrained_for_fsdp(args.pretrained_ckpt, model, rank)
        dist.barrier()

    # 获取参数分组（FSDP 包装前）
    _pg = model.get_param_groups()
    _neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron'}
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]

    # 保存原始模型引用
    raw_model = model

    # TensorBoard 看板（FSDP 包装前初始化，use_orig_params=True 保证引用有效）
    if args.dashboard_dir:
        from dashboard import SNNDashboard
        dashboard = SNNDashboard(args.dashboard_dir, raw_model, rank)
    else:
        dashboard = None

    # FSDP 包装
    model, device = wrap_model_fsdp(model, args, local_rank)

    # ==================== 数据 ====================
    train_ds = SFTDataset(args.sft_data_path, tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # ==================== 优化器 ====================
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': args.learning_rate, 'lr_mult': 1.0,
         'weight_decay': 0.01},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult), 'weight_decay': 0.0},
    ])

    # 恢复 SFT checkpoint
    tokens_seen = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch, best_loss, tokens_seen = load_checkpoint_fsdp(
            args.resume, model, optimizer, device, rank,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps * world_size

    Logger(f"\n{'='*60}", rank)
    Logger(f"SNN Language Model SFT (FSDP, {world_size} GPUs)", rank)
    Logger(f"  Vocab:       {args.vocab_size}", rank)
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}", rank)
    Logger(f"  SFT Data:    {args.sft_data_path}", rank)
    Logger(f"  Samples:     {len(train_ds):,}", rank)
    Logger(f"  Max length:  {args.max_length}", rank)
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} gpus × accum {args.accumulation_steps} = {effective_batch} effective", rank)
    Logger(f"  Epochs:      {args.epochs}", rank)
    Logger(f"  Steps/epoch: {iter_per_epoch:,}", rank)
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})", rank)
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)", rank)
    Logger(f"  Grad clip:   {args.grad_clip}", rank)
    Logger(f"  Precision:   bfloat16 (FSDP MixedPrecision)", rank)
    Logger(f"  Sharding:    {args.sharding_strategy}", rank)
    Logger(f"  no_sync:     accumulation_steps={args.accumulation_steps} (消除 {args.accumulation_steps-1} 次冗余通信)", rank)
    Logger(f"  Save every:  {args.save_interval} steps", rank)
    if args.pretrained_ckpt:
        Logger(f"  Pretrained:  {args.pretrained_ckpt}", rank)
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {local_rank})", rank)
    Logger(f"{'='*60}\n", rank)

    # ==================== 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tokens_seen = train_epoch(
            epoch, model, raw_model, train_loader, sampler, optimizer, ctx, args,
            iter_per_epoch, tokens_seen, rank, world_size, dashboard=dashboard,
        )

    # 最终保存
    Logger(f"\nSFT finished. Total tokens seen: {tokens_seen:,}", rank)
    if is_main_process(rank):
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
    save_checkpoint_fsdp(args.save_dir, model, optimizer,
                         args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen, rank)

    if dashboard:
        dashboard.close()

    cleanup_distributed()
