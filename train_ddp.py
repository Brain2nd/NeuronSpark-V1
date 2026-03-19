"""
分布式训练脚本：SNN 语言模型预训练（DDP 多卡并行）

基于 train.py 单卡脚本，使用 PyTorch DistributedDataParallel 实现数据并行。
训练逻辑、模型架构、超参数与单卡版完全一致。

用法：
    # 单机多卡（例如 4 张 GPU）
    torchrun --nproc_per_node=4 train_ddp.py \
        --D 768 --D_ff 2304 --batch_size 2 --accumulation_steps 8

    # 多机多卡（例如 2 机 × 4 卡）
    torchrun --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
        --nproc_per_node=4 train_ddp.py --D 768 --D_ff 2304

    # 单卡也能跑（等价于 train.py）
    torchrun --nproc_per_node=1 train_ddp.py --D 768 --D_ff 2304

    # 断续训练
    torchrun --nproc_per_node=4 train_ddp.py --resume checkpoints/latest.pt
"""

import os
import time
import math
import argparse
import warnings

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model import SNNLanguageModel
from dataset import PretrainDataset
from checkpoint_utils import save_checkpoint, load_checkpoint, load_model_weights

warnings.filterwarnings('ignore')


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
# 初始化
# ============================================================

def init_model(args, local_rank, rank):
    """初始化模型和分词器。"""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = SNNLanguageModel(
        vocab_size=args.vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
        activation_mode=args.activation_mode,
    )

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device=device, dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'SNN LM 总参数量：{total_params / 1e6:.3f} 百万', rank)

    return model, tokenizer, device


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, train_loader, sampler, optimizer, ctx, args,
                iter_per_epoch, tokens_seen, rank, world_size, dashboard=None,
                start_step=0):
    """训练一个 epoch（DDP 版本）。"""
    # 设置 sampler 的 epoch 以保证每个 epoch 的 shuffle 不同
    sampler.set_epoch(epoch)

    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        global_step = start_step + step
        X = X.to(f"cuda:{local_rank}")
        Y = Y.to(f"cuda:{local_rank}")
        loss_mask = loss_mask.to(f"cuda:{local_rank}")

        # 学习率调度（用 global_step 保证 resume 后 lr 连续）
        lr = get_lr(global_step, args.epochs * iter_per_epoch,
                     args.learning_rate, args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        # no_sync: 非边界步跳过 all-reduce，节省通信
        is_boundary = (step + 1) % args.accumulation_steps == 0
        sync_ctx = nullcontext() if is_boundary else model.no_sync()

        with sync_ctx:
            # 前向传播
            with ctx:
                out = model(X, Y)
                loss = out.last_loss / args.accumulation_steps
                loss_mask_flat = loss_mask.view(-1)
                loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

            # 反向传播（bf16 不需要 GradScaler）
            loss.backward()

        # 梯度累积到边界步时更新
        if is_boundary:
            # Natural Gradient: 补偿 b_beta/b_alpha 的 sigmoid/softplus 梯度衰减
            raw_model = model.module if isinstance(model, DDP) else model
            raw_model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Dashboard
            if dashboard and is_main_process(rank):
                raw_model = model.module if isinstance(model, DDP) else model
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
                }, raw_model, log_params=(step % args.log_interval == 0))

            optimizer.zero_grad(set_to_none=True)

        # 有效 token 数（汇总所有卡）
        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        # 日志（仅主进程）
        if global_step % args.log_interval == 0 and is_main_process(rank):
            batch_loss = loss.item() * args.accumulation_steps
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{}'.format(
                    epoch + 1, args.epochs, global_step, iter_per_epoch,
                    batch_loss, batch_ppl, lr, tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

        # 定期保存（仅主进程，带步数，保留最新 5 个）
        if (global_step + 1) % args.save_interval == 0:
            if is_main_process(rank):
                model.eval()
                save_checkpoint(args.save_dir, model, optimizer, None,
                                global_step + 1, epoch, batch_loss, tokens_seen)
                if dashboard:
                    raw_model = model.module if isinstance(model, DDP) else model
                    dashboard.log_save_point(global_step, raw_model)
                model.train()
            if world_size > 1:
                dist.barrier()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model Pretraining (DDP)")

    # SNN 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=24)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)
    parser.add_argument('--activation_mode', type=str, default='v2', choices=['v1', 'v2'],
                        help='激活模式: v1=V_post, v2=(1-β)·V_post (默认 v2)')

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8, help="每卡 batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=0)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 数据
    parser.add_argument("--data_path", type=str,
                        default="data/seq-monkey/seq_monkey_datawhale.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    # TensorBoard 看板
    parser.add_argument('--dashboard_dir', type=str, default=None,
                        help="TensorBoard 日志目录，例如 runs/pretrain")

    args = parser.parse_args()

    # ==================== 分布式初始化 ====================
    local_rank, rank, world_size = setup_distributed()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # 种子：每卡不同（保证数据不同），但可复现
    torch.manual_seed(42 + rank)

    # bf16 autocast（模型已是 bf16，autocast 确保中间计算也是 bf16）
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # ==================== 模型初始化 ====================
    model, tokenizer, device = init_model(args, local_rank, rank)

    # 恢复 checkpoint（DDP 包装前加载模型权重，确保参数引用一致）
    tokens_seen = 0
    start_epoch = 0
    start_step = 0
    _resume_optim_state = None
    if args.resume:
        # 加载模型权重（DDP 包装前，参数 in-place）
        load_model_weights(args.resume, model, device)
        # 读取训练状态（step/epoch/tokens），optimizer state 暂存
        if os.path.isdir(args.resume):
            _ts = torch.load(os.path.join(args.resume, 'training_state.pth'), map_location=device, weights_only=False)
        else:
            _ts = torch.load(args.resume, map_location=device, weights_only=False)
        start_step = _ts.get('step', 0)
        start_epoch = _ts.get('epoch', 0)
        tokens_seen = _ts.get('tokens_seen', 0)
        _resume_optim_state = _ts.get('optimizer_state', None)
        Logger(f"  Resumed model: step={start_step}, epoch={start_epoch}, tokens={tokens_seen:,}", rank)

    # DDP 包装（gradient_as_bucket_view 复用通信 buffer 省显存）
    model = DDP(model, device_ids=[local_rank],
                gradient_as_bucket_view=True)

    # ==================== 数据加载 ====================
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)

    # DistributedSampler 自动按 rank 划分数据
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # ==================== 优化器 ====================
    # 通过 .module 访问原始模型方法
    _pg = model.module.get_param_groups()
    # tuple 保证固定遍历顺序（set 在不同进程间顺序随机，导致 optimizer state 映射错乱）
    _neuron_keys = ('input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron')
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.learning_rate, 'lr_mult': 1.0},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult)},
    ])

    # 恢复 optimizer state（optimizer 创建后加载，参数引用已绑定）
    if _resume_optim_state is not None:
        optimizer.load_state_dict(_resume_optim_state)
        Logger("  Optimizer state restored.", rank)
    elif args.resume:
        Logger("  Optimizer state not found, starting fresh.", rank)

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps * world_size

    Logger(f"\n{'='*60}", rank)
    Logger(f"SNN Language Model Pretraining (DDP, {world_size} GPUs)", rank)
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
    Logger(f"  Precision:   bfloat16", rank)
    Logger(f"  Save every:  {args.save_interval} steps", rank)
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {local_rank})", rank)
    Logger(f"{'='*60}\n", rank)

    # ==================== TensorBoard 看板 ====================
    if args.dashboard_dir:
        from dashboard import SNNDashboard
        raw_model = model.module if isinstance(model, DDP) else model
        dashboard = SNNDashboard(args.dashboard_dir, raw_model, rank)
    else:
        dashboard = None

    # ==================== 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tokens_seen = train_epoch(
            epoch, model, train_loader, sampler, optimizer, ctx, args,
            iter_per_epoch, tokens_seen, rank, world_size, dashboard=dashboard,
            start_step=start_step,
        )

    if dashboard:
        dashboard.close()

    # 最终保存
    if is_main_process(rank):
        Logger(f"\nTraining finished. Total tokens seen: {tokens_seen:,}", rank)
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
        save_checkpoint(args.save_dir, model, optimizer, None,
                        args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen)

    cleanup_distributed()
