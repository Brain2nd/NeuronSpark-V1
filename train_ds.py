"""
分布式训练脚本：SNN 语言模型预训练（DeepSpeed ZeRO-2）

基于 train_ddp.py，使用 DeepSpeed ZeRO-2 实现优化器/梯度分片。
相比 DDP：优化器状态分片到 8 卡，每卡省 ~10GB 显存。

用法：
    deepspeed --num_gpus=8 train_ds.py \
        --D 1280 --N 8 --K 12 --num_layers 32 --D_ff 3840 \
        --vocab_size 64000 --batch_size 1 --accumulation_steps 100 \
        --deepspeed_config ds_config.json
"""

import os
import time
import math
import argparse
import warnings
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from transformers import AutoTokenizer

from model import SNNLanguageModel
from dataset import PretrainDataset
from checkpoint_utils import save_checkpoint, load_model_weights

warnings.filterwarnings('ignore')


# ============================================================
# 工具
# ============================================================

def is_main_process():
    if not dist.is_initialized():
        return int(os.environ.get("RANK", "0")) == 0
    return dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

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
# 初始化
# ============================================================

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

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

    # 神经元参数提升到 fp32
    for name, param in model.named_parameters():
        if name.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th')):
            param.data = param.data.float()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_count = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)
    Logger(f'SNN LM 总参数量：{total_params / 1e6:.3f} 百万 (fp32: {fp32_count / 1e6:.3f}M)')

    return model, tokenizer, device


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model_engine, train_loader, sampler, args,
                iter_per_epoch, tokens_seen, dashboard=None, start_step=0):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    sampler.set_epoch(epoch)

    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if step < start_step:
            if step % 10000 == 0 and rank == 0:
                print(f"  Skipping to step {start_step}... ({step}/{start_step})")
            continue
        global_step = step

        X = X.to(f"cuda:{local_rank}")
        Y = Y.to(f"cuda:{local_rank}")
        loss_mask = loss_mask.to(f"cuda:{local_rank}")

        # 学习率调度
        lr = get_lr(global_step, args.epochs * iter_per_epoch,
                     args.learning_rate, args.warmup_iters)
        for param_group in model_engine.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        # 前向传播
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model_engine(X, Y)
            loss = out.last_loss
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        # 反向传播 (DeepSpeed 自动处理梯度累积和通信)
        model_engine.backward(loss)

        # accumulation 边界：补偿梯度 + step
        is_boundary = (step + 1) % args.accumulation_steps == 0
        if is_boundary:
            model_engine.module.compensate_modulation_gradients()
        model_engine.step()

        if is_boundary and dashboard and is_main_process():
            raw_model = model_engine.module
            batch_loss = loss.item()
            # 梯度已被 step() 清零，跳过 grad 相关记录
            spend_time = time.time() - start_time
            dashboard.log_step(global_step, {
                'loss': batch_loss,
                'ppl': math.exp(min(batch_loss, 20.0)),
                'lr': lr,
                'tps': tokens_seen / spend_time if spend_time > 0 else 0,
                'tokens_seen': tokens_seen,
                'ponder_cost': out.ponder_cost.item() if out.ponder_cost is not None else 0.0,
                'memory_current_gb': torch.cuda.memory_allocated() / 1e9,
                'memory_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
            }, raw_model, log_params=(global_step % args.log_interval == 0))

        # 有效 token 数
        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        # 日志
        if global_step % args.log_interval == 0 and is_main_process():
            batch_loss = loss.item()
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

        # 定期保存
        if (global_step + 1) % args.save_interval == 0:
            if is_main_process():
                model_engine.module.eval()
                save_checkpoint(args.save_dir, model_engine.module,
                                model_engine.optimizer, None,
                                global_step + 1, epoch, loss.item(), tokens_seen)
                if dashboard:
                    dashboard.log_save_point(global_step, model_engine.module)
                model_engine.module.train()
            dist.barrier()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model Pretraining (DeepSpeed)")

    # SNN 模型参数
    parser.add_argument('--vocab_size', type=int, default=64000)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=24)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)
    parser.add_argument('--activation_mode', type=str, default='v2', choices=['v1', 'v2'])

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="每卡 micro batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--accumulation_steps', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 数据
    parser.add_argument("--data_path", type=str, default="data/pretrain_mix/dataset")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    # TensorBoard 看板
    parser.add_argument('--dashboard_dir', type=str, default=None)

    # DeepSpeed
    parser.add_argument('--local_rank', type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # ==================== 模型初始化 ====================
    model, tokenizer, device = init_model(args)

    # Resume
    tokens_seen = 0
    start_step = 0
    if args.resume:
        load_model_weights(args.resume, model, device)
        if os.path.isdir(args.resume):
            _ts = torch.load(os.path.join(args.resume, 'training_state.pth'),
                             map_location=device, weights_only=False)
        else:
            _ts = torch.load(args.resume, map_location=device, weights_only=False)
        start_step = _ts.get('step', 0)
        tokens_seen = _ts.get('tokens_seen', 0)
        Logger(f"  Resumed: step={start_step}, tokens={tokens_seen:,}")

    # ==================== 优化器参数组 ====================
    _pg = model.get_param_groups()
    _neuron_keys = ('input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron')
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.learning_rate, 'lr_mult': 1.0},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult)},
    ])

    # ==================== DeepSpeed 初始化 ====================
    # 读取 ds_config 并覆盖 CLI 参数
    ds_config_path = getattr(args, 'deepspeed_config', 'ds_config.json')
    with open(ds_config_path) as f:
        ds_config = json.load(f)
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['gradient_accumulation_steps'] = args.accumulation_steps
    ds_config['gradient_clipping'] = args.grad_clip

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    # ==================== 数据加载 ====================
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    iter_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps * world_size

    Logger(f"\n{'='*60}")
    Logger(f"SNN Language Model Pretraining (DeepSpeed ZeRO-2, {world_size} GPUs)")
    Logger(f"  Vocab:       {args.vocab_size}")
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}")
    Logger(f"  Data:        {args.data_path}")
    Logger(f"  Samples:     {len(train_ds):,}")
    Logger(f"  Max length:  {args.max_length}")
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} gpus × accum {args.accumulation_steps} = {effective_batch} effective")
    Logger(f"  Epochs:      {args.epochs}")
    Logger(f"  Steps/epoch: {iter_per_epoch:,}")
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})")
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)")
    Logger(f"  Grad clip:   {args.grad_clip}")
    Logger(f"  Precision:   bfloat16 + ZeRO-2")
    Logger(f"  Save every:  {args.save_interval} steps")
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {rank})")
    Logger(f"{'='*60}\n")

    # ==================== TensorBoard 看板 ====================
    if args.dashboard_dir:
        from dashboard import SNNDashboard
        dashboard = SNNDashboard(args.dashboard_dir, model_engine.module, rank)
    else:
        dashboard = None

    # ==================== 训练 ====================
    for epoch in range(args.epochs):
        tokens_seen = train_epoch(
            epoch, model_engine, train_loader, sampler, args,
            iter_per_epoch, tokens_seen, dashboard, start_step,
        )
        start_step = 0

    Logger("Training complete.")
    deepspeed.comm.destroy_process_group()
