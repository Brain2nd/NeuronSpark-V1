"""
SFT 训练脚本：SNN 语言模型监督微调（单卡）

基于 train.py 预训练脚本，核心差异：
  1. 使用 SFTDataset（对话格式 + loss mask 仅计算 assistant 回复）
  2. 加载预训练 checkpoint 权重（--pretrained_ckpt）
  3. Optimizer 状态不继承（微调从头开始优化）

数据格式：JSONL，每行一个 JSON list（由 deal_dataset.py 生成），格式如下：
  [{"role": "system", "content": "你是一个AI助手"}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
  tokenizer 必须配置 chat_template（ChatML 格式，已内置于 tokenizer_config.json）。

用法：
    conda activate SNN

    python sft.py \
        --pretrained_ckpt checkpoints/ckpt_step10000.pth \
        --sft_data_path data/sft/sft_data.jsonl \
        --batch_size 4 --accumulation_steps 16

    # 断续训练（从 SFT checkpoint 恢复）
    python sft.py --resume checkpoints_sft/ckpt_step500.pth
"""

import os
import time
import math
import argparse
import warnings

import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from model import SNNLanguageModel
from dataset import SFTDataset
from checkpoint_utils import save_checkpoint, load_checkpoint, load_model_weights

warnings.filterwarnings('ignore')


def Logger(content):
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

def init_model(args):
    """初始化模型和分词器。"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    model = model.to(args.device)
    Logger(f'SNN LM 总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, train_loader, optimizer, scaler, ctx, args, iter_per_epoch, tokens_seen,
                dashboard=None):
    """训练一个 epoch（SFT 版本，与预训练逻辑完全一致）。"""
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 学习率调度
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch,
                     args.learning_rate, args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        # 前向传播
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Dashboard
            if dashboard:
                batch_loss_db = loss.item() * args.accumulation_steps
                spend_time_db = time.time() - start_time
                dashboard.log_step(step, {
                    'loss': batch_loss_db,
                    'ppl': math.exp(min(batch_loss_db, 20.0)),
                    'lr': lr,
                    'tps': tokens_seen / spend_time_db if spend_time_db > 0 else 0,
                    'tokens_seen': tokens_seen,
                    'memory_current_gb': torch.cuda.memory_allocated() / 1e9 if args.device != 'cpu' else 0,
                    'memory_peak_gb': torch.cuda.max_memory_allocated() / 1e9 if args.device != 'cpu' else 0,
                }, model, log_params=(step % args.log_interval == 0))

            optimizer.zero_grad(set_to_none=True)

        valid_tokens = int(loss_mask_flat.sum().item())
        tokens_seen += valid_tokens

        # 日志
        if step % args.log_interval == 0:
            batch_loss = loss.item() * args.accumulation_steps
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_str = ""
            if args.device != 'cpu':
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" | Mem {mem_cur:.1f}/{mem_peak:.1f}GB"
            Logger(
                'SFT Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f} epoch_Time:{}min{}'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, batch_ppl, lr, tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_str))

        # 定期保存
        if (step + 1) % args.save_interval == 0:
            model.eval()
            global_step = epoch * iter_per_epoch + step + 1
            save_checkpoint(args.save_dir, model, optimizer, scaler,
                            global_step, epoch, batch_loss, tokens_seen)
            if dashboard:
                dashboard.log_save_point(global_step, model)
            model.train()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model SFT (Supervised Fine-Tuning)")

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

    # SFT 特有参数
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='预训练 checkpoint 路径（首次 SFT 必须指定）')
    parser.add_argument('--sft_data_path', type=str,
                        default='data/sft/sft_data.jsonl',
                        help='SFT 对话数据路径（JSONL，ChatML 格式）')

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints_sft")
    parser.add_argument("--epochs", type=int, default=3, help="SFT 训练轮数（通常 1-5）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)

    # 优化参数（SFT 默认更低的 lr）
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率（SFT 通常比预训练低 2-10x）')
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # Tokenizer
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/")

    # 断续训练
    parser.add_argument('--resume', type=str, default=None, help='从 SFT checkpoint 恢复')

    # TensorBoard 看板
    parser.add_argument('--dashboard_dir', type=str, default=None,
                        help="TensorBoard 日志目录，例如 runs/sft")

    args = parser.parse_args()

    # ==================== 环境 ====================
    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        'cuda', dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)

    # ==================== 模型初始化 ====================
    model, tokenizer = init_model(args)

    # 加载预训练权重
    if args.pretrained_ckpt and not args.resume:
        load_model_weights(args.pretrained_ckpt, model, args.device)

    # ==================== SFT 数据 ====================
    train_ds = SFTDataset(args.sft_data_path, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ==================== 优化器 ====================
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    _pg = model.get_param_groups()
    _neuron_keys = ('input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron')
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]
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
        start_step, start_epoch, best_loss, tokens_seen = load_checkpoint(
            args.resume, model, optimizer, scaler, args.device,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)

    Logger(f"\n{'='*60}")
    Logger(f"SNN Language Model SFT (Supervised Fine-Tuning)")
    Logger(f"  Vocab:       {args.vocab_size}")
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}")
    Logger(f"  SFT Data:    {args.sft_data_path}")
    Logger(f"  Samples:     {len(train_ds):,}")
    Logger(f"  Max length:  {args.max_length}")
    Logger(f"  Batch size:  {args.batch_size} × accum {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} effective")
    Logger(f"  Epochs:      {args.epochs}")
    Logger(f"  Steps/epoch: {iter_per_epoch:,}")
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})")
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)")
    Logger(f"  Grad clip:   {args.grad_clip}")
    Logger(f"  Precision:   {args.dtype}")
    Logger(f"  Save every:  {args.save_interval} steps")
    if args.pretrained_ckpt:
        Logger(f"  Pretrained:  {args.pretrained_ckpt}")
    if args.device != 'cpu':
        mem_baseline = torch.cuda.memory_allocated() / 1e9
        Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline")
    Logger(f"{'='*60}\n")

    # ==================== TensorBoard 看板 ====================
    if args.dashboard_dir:
        from dashboard import SNNDashboard
        dashboard = SNNDashboard(args.dashboard_dir, model)
    else:
        dashboard = None

    # ==================== 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        tokens_seen = train_epoch(
            epoch, model, train_loader, optimizer, scaler, ctx, args,
            iter_per_epoch, tokens_seen, dashboard=dashboard,
        )

    if dashboard:
        dashboard.close()

    Logger(f"\nSFT finished. Total tokens seen: {tokens_seen:,}")
    if args.device != 'cpu':
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    save_checkpoint(args.save_dir, model, optimizer, scaler,
                    args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen)
