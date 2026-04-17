"""
分布式预训练脚本：SNN 语言模型，NeuronLangevin 优化器 (DeepSpeed stage=0)

与 train_ds.py 的核心差异:
  1. 优化器换成 NeuronLangevin (ortho drift + Langevin diffusion + AdamW)
  2. DeepSpeed 配置强制 zero_optimization.stage = 0
     原因: 正交化 drift (Newton-Schulz) 需要完整的 2D momentum 矩阵;
           ZeRO-1/2/3 对 optimizer state 做 flatten+partition 会破坏 2D 形状;
           stage=0 用 DeepSpeed 的 bf16 管理 + 梯度累积, 不分片 optimizer state.
  3. halt_proj 保留 bf16: ortho drift 每步 ~lr 量级远大于 bf16 quantum, 无需 fp32 master;
     且 halt fp32 与 autocast bf16 输入 matmul 报 dtype mismatch.

用法 (新起):
    deepspeed --num_gpus=8 train_ds_neuron.py \
        --data_path data/pretrain_mix/dataset \
        --vocab_size 64000 --D 1024 --N 8 --K 12 --num_layers 24 --D_ff 3072 \
        --batch_size 1 --accumulation_steps 100 \
        --adamw_lr 2e-4 --ortho_lr 2e-3 \
        --T_halt 1e-3 --T_ortho 5e-5 --T_plif 1e-4 \
        --dashboard_dir runs/train_neuron_<tag>

用法 (继续预训练 / 退火):
    # --pretrained_ckpt: 只加载权重, optimizer 从零 (推荐, 避免 Adam → NeuronLangevin 状态迁移)
    deepspeed --num_gpus=8 train_ds_neuron.py \
        --pretrained_ckpt checkpoints/ckpt_step479000 \
        --adamw_lr 5e-5 --ortho_lr 1e-3 \
        ...

    # --resume: 加载权重 + step/tokens_seen (optimizer 也从零, 因为 NeuronLangevin 和 Adam 状态不兼容)
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
from atomic_ops.neuron_langevin import NeuronLangevin, build_param_groups

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


def get_lr(it, total_iters, peak_lr, warmup_iters):
    """Cosine with warmup. 返回当前 peak_lr 下的 base lr 值 (不含 lr_mult, 由 NeuronLangevin 应用)."""
    min_lr = peak_lr / 10
    if it < warmup_iters:
        return peak_lr * it / warmup_iters
    if it > total_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


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
    )

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device=device, dtype=torch.bfloat16)

    # fp32 白名单: 仅 PLIF/SNN 神经元 1D 参数 (与 train_ds.py 一致).
    # halt_proj 保留 bf16: ortho drift 的 lr=2e-3 量级 >> bf16 quantum 1.6e-4, 不需要 fp32 master.
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

        # 每组独立 cosine schedule: 更新 pg['lr'] (不含 lr_mult, 由 NeuronLangevin 内部乘)
        for pg in model_engine.optimizer.param_groups:
            peak = pg['peak_lr']
            pg['lr'] = get_lr(global_step, args.epochs * iter_per_epoch,
                              peak, args.warmup_iters)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model_engine(X, Y)
            loss = out.last_loss
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        model_engine.backward(loss)

        is_boundary = (step + 1) % args.accumulation_steps == 0
        if is_boundary:
            model_engine.module.compensate_modulation_gradients()
        model_engine.step()

        if is_boundary and dashboard and is_main_process():
            raw_model = model_engine.module
            batch_loss = loss.item()
            spend_time = time.time() - start_time
            lr_adamw = next(
                (pg['lr'] for pg in model_engine.optimizer.param_groups
                 if pg.get('tag') == 'adamw_other'),
                0.0,
            )
            dashboard.log_step(global_step, {
                'loss': batch_loss,
                'ppl': math.exp(min(batch_loss, 20.0)),
                'lr': lr_adamw,
                'tps': tokens_seen / spend_time if spend_time > 0 else 0,
                'tokens_seen': tokens_seen,
                'ponder_cost': out.ponder_cost.item() if out.ponder_cost is not None else 0.0,
                'memory_current_gb': torch.cuda.memory_allocated() / 1e9,
                'memory_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
            }, raw_model, log_params=(global_step % args.log_interval == 0))

        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        if global_step % args.log_interval == 0 and is_main_process():
            batch_loss = loss.item()
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            lr_adamw = next(
                (pg['lr'] for pg in model_engine.optimizer.param_groups
                 if pg.get('tag') == 'adamw_other'),
                0.0,
            )
            lr_ortho = next(
                (pg['lr'] for pg in model_engine.optimizer.param_groups
                 if pg.get('tag') == 'ortho_matrix'),
                0.0,
            )
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} '
                'lr_a:{:.7f} lr_o:{:.7f} TPS:{:.0f} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{}'.format(
                    epoch + 1, args.epochs, global_step, iter_per_epoch,
                    batch_loss, batch_ppl, lr_adamw, lr_ortho, tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

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
    parser = argparse.ArgumentParser(
        description="SNN LM Pretraining (NeuronLangevin: ortho drift + Langevin diffusion)"
    )

    # 模型
    parser.add_argument('--vocab_size', type=int, default=64000)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=24)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)

    # 训练
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument('--accumulation_steps', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=500)

    # NeuronLangevin 优化器: drift 超参
    parser.add_argument('--adamw_lr', type=float, default=2e-4,
                        help="AdamW 组 peak lr (embed/norm/conv1d/小矩阵 + PLIF neuron×plif_lr_mult)")
    parser.add_argument('--ortho_lr', type=float, default=2e-3,
                        help="ortho 组 peak lr (矩阵 ortho_matrix + halt_proj)")
    parser.add_argument('--plif_lr_mult', type=float, default=10.0,
                        help="PLIF neuron 组对 adamw_lr 的倍率 (沿用预训练 ×10)")
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.95,
                        help="ortho drift 的 momentum β")
    parser.add_argument('--ns_steps', type=int, default=5,
                        help="Newton-Schulz 迭代步数")
    parser.add_argument('--ortho_min_dim', type=int, default=8,
                        help="小于该最小维的矩阵 fallback 到 AdamW (避免 NS 不稳)")

    # Langevin 温度 (参数空间 Gaussian 噪声)
    parser.add_argument('--T_halt', type=float, default=1e-3,
                        help="halt_proj 的 Langevin 温度 (推高 halt 探索)")
    parser.add_argument('--T_ortho', type=float, default=5e-5,
                        help="其它 ortho 矩阵的 Langevin 温度")
    parser.add_argument('--T_plif', type=float, default=1e-4,
                        help="PLIF 神经元参数的 Langevin 温度")
    parser.add_argument('--T_adamw_other', type=float, default=0.0,
                        help="embed/norm/conv1d 的 Langevin 温度 (通常 0)")

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default="data/pretrain_mix/dataset")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='加载权重 + step/tokens; optimizer 从零 (NeuronLangevin 和旧 Adam 状态不兼容)')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='只加载权重, step=0, optimizer 从零')

    # 看板
    parser.add_argument('--dashboard_dir', type=str, default=None)

    # DeepSpeed
    parser.add_argument('--local_rank', type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # ==================== 模型初始化 ====================
    model, tokenizer, device = init_model(args)

    # Resume / Pretrained
    tokens_seen = 0
    start_step = 0
    if args.pretrained_ckpt:
        load_model_weights(args.pretrained_ckpt, model, device)
        Logger(f"  Loaded pretrained weights: {args.pretrained_ckpt} (step=0, fresh optimizer)")
    elif args.resume:
        load_model_weights(args.resume, model, device)
        if os.path.isdir(args.resume):
            _ts = torch.load(os.path.join(args.resume, 'training_state.pth'),
                             map_location=device, weights_only=False)
        else:
            _ts = torch.load(args.resume, map_location=device, weights_only=False)
        start_step = _ts.get('step', 0)
        tokens_seen = _ts.get('tokens_seen', 0)
        Logger(f"  Resumed: step={start_step}, tokens={tokens_seen:,} "
               f"(optimizer state 不迁移, 从零重新累积 momentum)")

    # ==================== 优化器参数组 ====================
    param_groups = build_param_groups(
        model,
        ortho_lr=args.ortho_lr,
        adamw_lr=args.adamw_lr,
        plif_lr_mult=args.plif_lr_mult,
        weight_decay=args.weight_decay,
        T_halt=args.T_halt,
        T_ortho=args.T_ortho,
        T_plif=args.T_plif,
        T_adamw_other=args.T_adamw_other,
        momentum=args.momentum,
        ortho_min_dim=args.ortho_min_dim,
        verbose=is_main_process(),
    )
    # peak_lr = 原始基础 lr (不含 lr_mult); 训练 loop 里 cosine schedule 动 pg['lr']
    for pg in param_groups:
        pg['peak_lr'] = pg['lr']

    if is_main_process():
        print("\n=== NeuronLangevin param groups ===")
        for pg in param_groups:
            n_p = sum(p.numel() for p in pg['params'])
            effective_peak = pg['peak_lr'] * pg.get('lr_mult', 1.0)
            print(f"  [{pg['tag']:15s}] tensors={len(pg['params']):3d}  params={n_p:>12,}  "
                  f"optim={pg['optim_type']:5s}  peak_lr={pg['peak_lr']:.2e} ×{pg.get('lr_mult', 1.0):.1f} = {effective_peak:.2e}  "
                  f"T={pg['temperature']:.2e}")
        print()

    optimizer = NeuronLangevin(param_groups)

    # ==================== DeepSpeed 初始化 ====================
    ds_config_path = getattr(args, 'deepspeed_config', None) or 'ds_config.json'
    with open(ds_config_path) as f:
        ds_config = json.load(f)
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['gradient_accumulation_steps'] = args.accumulation_steps
    ds_config['gradient_clipping'] = args.grad_clip
    # 强制 stage=0: NeuronLangevin 的 ortho drift 需完整 2D momentum
    ds_config.setdefault('zero_optimization', {})
    ds_config['zero_optimization']['stage'] = 0
    ds_config['zero_allow_untested_optimizer'] = True

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

    Logger(f"\n{'='*70}")
    Logger(f"SNN LM Pretraining (NeuronLangevin, DeepSpeed stage=0, {world_size} GPUs)")
    Logger(f"  Vocab:       {args.vocab_size}")
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}")
    Logger(f"  Data:        {args.data_path}")
    Logger(f"  Samples:     {len(train_ds):,}")
    Logger(f"  Max length:  {args.max_length}")
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} gpus × accum {args.accumulation_steps} = {effective_batch} effective")
    Logger(f"  Epochs:      {args.epochs}")
    Logger(f"  Steps/epoch: {iter_per_epoch:,}")
    Logger(f"  Optim:       NeuronLangevin")
    Logger(f"    adamw_lr={args.adamw_lr}, ortho_lr={args.ortho_lr}, plif_mult={args.plif_lr_mult}")
    Logger(f"    T_halt={args.T_halt}, T_ortho={args.T_ortho}, T_plif={args.T_plif}, T_other={args.T_adamw_other}")
    Logger(f"    momentum={args.momentum}, ns_steps={args.ns_steps}")
    Logger(f"  Warmup:      {args.warmup_iters} steps → cosine → peak/10")
    Logger(f"  Grad clip:   {args.grad_clip}")
    Logger(f"  Precision:   bf16 + fp32 master (白名单: PLIF neuron)")
    Logger(f"  Save every:  {args.save_interval} steps")
    if args.pretrained_ckpt:
        Logger(f"  Pretrained:  {args.pretrained_ckpt}")
    if args.resume:
        Logger(f"  Resume from: {args.resume}")
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {rank})")
    Logger(f"{'='*70}\n")

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
