"""
分布式训练脚本：SNN 语言模型预训练（FSDP 全分片数据并行）

相比 train_ddp.py 的核心改进：
  1. FSDP 全分片：参数/梯度/优化器状态按层分片，显存占用 ~1/N
  2. no_sync() 梯度累积：消除 accumulation 期间的冗余通信
  3. bf16 混合精度：参数 bf16，归约 fp32，无需 GradScaler
  4. 通信计算重叠：forward_prefetch + backward_prefetch
  5. Checkpoint 兼容：保存 FULL_STATE_DICT，可用 generate_sample.py 单卡加载

用法：
    # 单机多卡（例如 4 张 GPU）
    torchrun --nproc_per_node=4 train_fsdp.py \
        --D 1024 --D_ff 3072 --batch_size 2 --accumulation_steps 8

    # 多机多卡（例如 2 机 × 4 卡）
    torchrun --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
        --nproc_per_node=4 train_fsdp.py --D 1024 --D_ff 3072

    # 切换分片策略
    torchrun --nproc_per_node=4 train_fsdp.py --sharding_strategy hybrid_shard

    # 断续训练
    torchrun --nproc_per_node=4 train_fsdp.py --resume checkpoints/ckpt_step5000.pth
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
from dataset import PretrainDataset

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
    """初始化分布式环境（由 torchrun 自动设置环境变量）。"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_distributed():
    """清理分布式环境。"""
    dist.destroy_process_group()


def is_main_process(rank):
    """是否为主进程（rank 0）。"""
    return rank == 0


def Logger(content, rank=0):
    """仅主进程打印日志。"""
    if is_main_process(rank):
        print(content)


# ============================================================
# 学习率调度（与 train.py 一致）
# ============================================================

def get_lr(it, total_iters, learning_rate, warmup_iters):
    """余弦退火学习率调度。"""
    min_lr = learning_rate / 10

    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > total_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# Checkpoint（FSDP 专用：FULL_STATE_DICT 模式）
# ============================================================

def save_checkpoint_fsdp(save_dir, model, optimizer, step, epoch, best_loss, tokens_seen,
                         rank):
    """保存 FSDP checkpoint（FULL_STATE_DICT 模式，兼容单卡加载）。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'ckpt_step{step}.pth')

    # 收集完整模型状态到 rank 0 CPU
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()

    # 收集完整优化器状态
    full_optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if is_main_process(rank):
        # 从 FSDP 包装中获取模型配置
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

    dist.barrier()


def load_checkpoint_fsdp(path, model, optimizer, device, rank):
    """加载 checkpoint 恢复训练（所有 rank 加载 full state dict，FSDP 自动分片）。"""
    Logger(f"Loading checkpoint from {path}...", rank)

    # 所有 rank 加载完整 state dict
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # 加载模型权重
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'trainable_state_dict' in ckpt:
            model.load_state_dict(ckpt['trainable_state_dict'])

    # 加载优化器状态
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
    """初始化模型和分词器（FSDP 包装前）。"""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = SNNLanguageModel(
        vocab_size=args.vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
        ek_floor=args.ek_floor,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'SNN LM 总参数量：{total_params / 1e6:.3f} 百万', rank)

    return model, tokenizer


def wrap_model_fsdp(model, args, local_rank):
    """FSDP 包装模型。"""
    device = torch.device(f"cuda:{local_rank}")

    # 自动包装策略: 每个 SNNDecoderLayer (~56M 参数) 作为独立 FSDP 单元
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={SNNDecoderLayer},
    )

    # 混合精度: 参数 bf16，归约 fp32，buffer bf16
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    # 分片策略
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
                iter_per_epoch, tokens_seen, rank, world_size, dashboard=None,
                accum_step=0):
    """训练一个 epoch（FSDP 版本，含 no_sync 梯度累积）。"""
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

        # no_sync: 梯度累积期间跳过 allreduce，仅在边界步同步
        is_boundary = (step + 1) % args.accumulation_steps == 0
        sync_ctx = nullcontext() if is_boundary else model.no_sync()

        with sync_ctx:
            # 前向传播
            with ctx:
                out = model(X, Y)
                loss = out.last_loss / args.accumulation_steps
                loss_mask_flat = loss_mask.view(-1)
                loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()
                # 动态 K: ponder cost 正则化（鼓励用更少步数处理简单 token）
                if out.ponder_cost is not None and args.ponder_weight > 0:
                    loss = loss + args.ponder_weight * out.ponder_cost / args.accumulation_steps
                # E[K] 下界惩罚（遏制 PonderNet 坍缩: 深层 E[K] → 1 的死亡螺旋）
                if out.ek_floor_cost is not None and args.ek_floor_weight > 0:
                    loss = loss + args.ek_floor_weight * out.ek_floor_cost / args.accumulation_steps
                # SNVR: 层间权重范数方差正则化（遏制 Jacobian 谱范数发散）
                if out.snvr_cost is not None and args.snvr_weight > 0:
                    loss = loss + args.snvr_weight * out.snvr_cost / args.accumulation_steps

            # 反向传播（bf16 不需要 GradScaler）
            loss.backward()

        # 梯度累积到边界步时更新
        if is_boundary:
            # Natural Gradient: 补偿调制参数梯度衰减
            # use_orig_params=True 保证可直接访问原始参数梯度
            raw_model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            accum_step += 1

            # Dashboard: optimizer.step() 后、zero_grad() 前（梯度仍可用）
            # 用 accum_step（optimizer 步数）而非 dataloader step 判断频率，
            # 因为 boundary step ≡ accum_steps-1 (mod accum_steps)，
            # 与 log_interval 整除时条件永远无法满足
            need_log = accum_step % args.log_interval == 0
            need_save = accum_step % args.save_interval == 0
            if need_log or need_save:
                with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                    if dashboard:
                        if need_log:
                            batch_loss_db = loss.item() * args.accumulation_steps
                            spend_time_db = time.time() - start_time
                            dashboard.log_step(accum_step, {
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
                            dashboard.log_save_point(accum_step, raw_model)

            optimizer.zero_grad(set_to_none=True)

        # 有效 token 数（汇总所有卡）
        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        # 日志（仅主进程）
        if step % args.log_interval == 0 and is_main_process(rank):
            batch_loss = loss.item() * args.accumulation_steps
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            ponder_str = ""
            if out.ponder_cost is not None:
                pc_val = out.ponder_cost.item()
                ponder_str = f" pc:{pc_val:.2f}"
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f}{} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{} [FSDP]'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, batch_ppl, lr, tps, ponder_str,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

        # 定期保存（仅在梯度累积边界步，确保 optimizer 已更新）
        if is_boundary and accum_step % args.save_interval == 0:
            model.eval()
            cur_loss = loss.item() * args.accumulation_steps
            save_checkpoint_fsdp(args.save_dir, model, optimizer,
                                 accum_step, epoch, cur_loss, tokens_seen, rank)
            model.train()

    return tokens_seen, accum_step


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model Pretraining (FSDP)")

    # SNN 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8, help="每卡 batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="DataLoader prefetch_factor")

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='投影权重 weight decay（遏制 W_beta_x 等权重爆炸）')
    parser.add_argument('--ponder_weight', type=float, default=0.01,
                        help='动态 K ponder cost 正则化权重')
    parser.add_argument('--ek_floor', type=float, default=4.0,
                        help='E[K] 下界: 低于此值时产生惩罚（遏制 PonderNet 坍缩）')
    parser.add_argument('--ek_floor_weight', type=float, default=0.1,
                        help='E[K] 下界惩罚权重')
    parser.add_argument('--snvr_weight', type=float, default=0.01,
                        help='SNVR 层间权重范数方差正则化权重')

    # FSDP 参数
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=list(SHARDING_STRATEGIES.keys()),
                        help="FSDP 分片策略")

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)

    # 数据
    parser.add_argument("--data_path", type=str,
                        default="data/seq-monkey/seq_monkey_datawhale.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    # TensorBoard 看板（不传则禁用，零开销）
    parser.add_argument('--dashboard_dir', type=str, default=None,
                        help="TensorBoard 日志目录，例如 runs/pretrain")

    args = parser.parse_args()

    # ==================== 分布式初始化 ====================
    local_rank, rank, world_size = setup_distributed()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # 种子：每卡不同（保证数据不同），但可复现
    torch.manual_seed(42 + rank)

    # 性能优化
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # 混合精度上下文（与 FSDP MixedPrecision 配合）
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # ==================== 模型初始化 ====================
    model, tokenizer = init_model(args, local_rank, rank)

    # 获取参数分组（FSDP 包装前，保持对原始参数的引用）
    # 三组: decay（投影权重，加 weight decay 遏制权重爆炸）
    #       no_decay（归一化/halt/embedding，不加 weight decay）
    #       neuron（神经元参数，10× lr，不加 weight decay）
    _pg = model.get_param_groups()
    _neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron'}
    _no_decay_keys = {'rms_norms', 'norm', 'halt_projs', 'embedding'}
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    no_decay_params = [p for k in _no_decay_keys if k in _pg for p in _pg[k]]
    decay_params = [p for k, ps in _pg.items()
                    if k not in _neuron_keys and k not in _no_decay_keys
                    for p in ps]

    # 保存原始模型引用（用于 compensate_modulation_gradients）
    raw_model = model

    # TensorBoard 看板（FSDP 包装前初始化，use_orig_params=True 保证引用有效）
    if args.dashboard_dir:
        from dashboard import SNNDashboard
        dashboard = SNNDashboard(args.dashboard_dir, raw_model, rank)
    else:
        dashboard = None

    # FSDP 包装（sync_module_states=True 从 rank 0 广播初始权重）
    model, device = wrap_model_fsdp(model, args, local_rank)

    # ==================== 数据加载 ====================
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)
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

    # ==================== 优化器（AdamW: weight decay 遏制权重爆炸）====================
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': args.learning_rate, 'lr_mult': 1.0,
         'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'lr': args.learning_rate, 'lr_mult': 1.0,
         'weight_decay': 0.0},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult), 'weight_decay': 0.0},
    ])

    # 恢复 checkpoint
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
    Logger(f"SNN Language Model Pretraining (FSDP, {world_size} GPUs)", rank)
    Logger(f"  Vocab:       {args.vocab_size}", rank)
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}", rank)
    Logger(f"  Data:        {args.data_path}", rank)
    Logger(f"  Samples:     {len(train_ds):,}", rank)
    Logger(f"  Max length:  {args.max_length}", rank)
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} gpus × accum {args.accumulation_steps} = {effective_batch} effective", rank)
    Logger(f"  Epochs:      {args.epochs}", rank)
    Logger(f"  Steps/epoch: {iter_per_epoch:,}", rank)
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})", rank)
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)", rank)
    Logger(f"  Grad clip:   {args.grad_clip}", rank)
    Logger(f"  Weight decay:{args.weight_decay} (投影权重)", rank)
    Logger(f"  E[K] floor:  {args.ek_floor} (weight={args.ek_floor_weight})", rank)
    Logger(f"  Precision:   bfloat16 (FSDP MixedPrecision)", rank)
    Logger(f"  Sharding:    {args.sharding_strategy}", rank)
    Logger(f"  no_sync:     accumulation_steps={args.accumulation_steps} (消除 {args.accumulation_steps-1} 次冗余通信)", rank)
    Logger(f"  Save every:  {args.save_interval} steps", rank)
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {local_rank})", rank)
    Logger(f"{'='*60}\n", rank)

    # ==================== 训练 ====================
    accum_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tokens_seen, accum_step = train_epoch(
            epoch, model, raw_model, train_loader, sampler, optimizer, ctx, args,
            iter_per_epoch, tokens_seen, rank, world_size, dashboard=dashboard,
            accum_step=accum_step,
        )

    # 最终保存
    Logger(f"\nTraining finished. Total tokens seen: {tokens_seen:,}", rank)
    if is_main_process(rank):
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
    save_checkpoint_fsdp(args.save_dir, model, optimizer,
                         args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen, rank)

    if dashboard:
        dashboard.close()

    cleanup_distributed()
