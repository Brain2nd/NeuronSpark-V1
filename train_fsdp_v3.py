"""
分布式训练脚本: NeuronSpark v3 混合架构预训练 (FSDP)

v3 Deep Copy NVIDIA 变化 (对比旧 train_fsdp_v3.py):
  1. 模型: NeuronSparkV3ForCausalLM + NeuronSparkV3Config
  2. FSDP wrap: SparkBlock 作为独立单元
  3. Loss: CE + ponder + ek_floor + mtp (移除 load_balance_loss)
  4. 参数分组: decay/no_decay/neuron (匹配新模型结构)
  5. gradient compensation: 通过 model.compensate_modulation_gradients()

用法:
    torchrun --nproc_per_node=4 train_fsdp_v3.py \
        --D 1024 --N 8 --batch_size 2 --accumulation_steps 8
"""

import os
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

from model_v3 import NeuronSparkV3ForCausalLM, NeuronSparkV3Config, SparkBlock
from dataset import PretrainDataset

warnings.filterwarnings('ignore')

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
# WSD 学习率调度 (Warmup-Stable-Decay)
# ============================================================

def get_lr_wsd(it, total_iters, learning_rate, warmup_iters, stable_ratio=0.8):
    """WSD 学习率调度: warmup → stable → minus-sqrt decay。"""
    min_lr = learning_rate / 100

    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    stable_end = warmup_iters + int((total_iters - warmup_iters) * stable_ratio)

    if it <= stable_end:
        return learning_rate

    if it > total_iters:
        return min_lr

    decay_steps = total_iters - stable_end
    decay_progress = (it - stable_end) / decay_steps
    coeff = 1.0 - math.sqrt(decay_progress)
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint_fsdp(save_dir, model, optimizer, step, epoch, best_loss, tokens_seen,
                         rank, config):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'ckpt_step{step}.pth')

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()

    full_optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if is_main_process(rank):
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state': full_optim_state,
            'step': step,
            'epoch': epoch,
            'best_loss': best_loss,
            'tokens_seen': tokens_seen,
            'model_config': {
                'vocab_size': config.vocab_size,
                'hidden_size': config.hidden_size,
                'num_hidden_layers': config.num_hidden_layers,
                'ssm_N': config.ssm_N,
                'ssm_K': config.ssm_K,
                'ssm_v_th_min': config.ssm_v_th_min,
                'ssm_ek_floor': config.ssm_ek_floor,
                'num_attention_heads': config.num_attention_heads,
                'num_key_value_heads': config.num_key_value_heads,
                'head_dim': config.head_dim,
                'intermediate_size': config.intermediate_size,
                'n_routed_experts': config.n_routed_experts,
                'num_experts_per_tok': config.num_experts_per_tok,
                'moe_intermediate_size': config.moe_intermediate_size,
                'moe_shared_expert_intermediate_size': config.moe_shared_expert_intermediate_size,
                'routed_scaling_factor': config.routed_scaling_factor,
                'hybrid_override_pattern': config.hybrid_override_pattern,
                'n_mtp_heads': config.n_mtp_heads,
                'max_seq_len': config.max_seq_len,
            },
            'version': 'v3-deepcopy',
        }, path)
        print(f"  → Checkpoint saved: {path}")

    dist.barrier()


def load_checkpoint_fsdp(path, model, optimizer, device, rank):
    Logger(f"Loading checkpoint from {path}...", rank)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])

    if 'optimizer_state' in ckpt:
        try:
            full_optim_state = ckpt['optimizer_state']
            sharded_optim_state = FSDP.shard_full_optim_state_dict(full_optim_state, model)
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

def build_config(args):
    """从命令行参数构建 NeuronSparkV3Config。"""
    return NeuronSparkV3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.D,
        num_hidden_layers=args.num_layers,
        ssm_N=args.N,
        ssm_K=args.K,
        ssm_v_th_min=args.v_th_min,
        ssm_ek_floor=args.ek_floor,
        num_attention_heads=args.n_q_heads,
        num_key_value_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        n_routed_experts=args.num_experts,
        num_experts_per_tok=args.moe_top_k,
        moe_intermediate_size=args.expert_hidden,
        moe_shared_expert_intermediate_size=args.shared_expert_hidden,
        routed_scaling_factor=args.routed_scaling_factor,
        n_mtp_heads=args.n_mtp_heads,
        max_seq_len=args.max_length,
    )


def init_model(config, local_rank, rank):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = NeuronSparkV3ForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'v3 Deep Copy 总参数量: {total_params / 1e6:.3f}M', rank)

    return model, tokenizer


def wrap_model_fsdp(model, args, local_rank):
    device = torch.device(f"cuda:{local_rank}")

    # SparkBlock 作为独立 FSDP 单元
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={SparkBlock},
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
                iter_per_epoch, tokens_seen, rank, world_size, config, accum_step=0):
    sampler.set_epoch(epoch)
    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])
    total_iters = args.epochs * iter_per_epoch

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(f"cuda:{local_rank}", non_blocking=True)
        Y = Y.to(f"cuda:{local_rank}", non_blocking=True)
        loss_mask = loss_mask.to(f"cuda:{local_rank}", non_blocking=True)

        # WSD 学习率调度
        global_step = epoch * iter_per_epoch + step
        lr = get_lr_wsd(global_step, total_iters, args.learning_rate, args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        is_boundary = (step + 1) % args.accumulation_steps == 0
        sync_ctx = nullcontext() if is_boundary else model.no_sync()

        with sync_ctx:
            with ctx:
                out = model(X, Y)
                loss = out.last_loss / args.accumulation_steps
                loss_mask_flat = loss_mask.view(-1)
                loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

                # Ponder cost (BioSSM)
                if out.ponder_cost is not None and args.ponder_weight > 0:
                    loss = loss + args.ponder_weight * out.ponder_cost / args.accumulation_steps

                # E[K] floor (BioSSM)
                if out.ek_floor_cost is not None and args.ek_floor_weight > 0:
                    loss = loss + args.ek_floor_weight * out.ek_floor_cost / args.accumulation_steps

                # MTP loss
                if out.mtp_loss is not None and args.mtp_weight > 0:
                    loss = loss + args.mtp_weight * out.mtp_loss / args.accumulation_steps

            loss.backward()

        if is_boundary:
            raw_model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            accum_step += 1
            optimizer.zero_grad(set_to_none=True)

        # 有效 token 数
        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        # 日志
        if step % args.log_interval == 0 and is_main_process(rank):
            batch_loss = loss.item() * args.accumulation_steps
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            extra = ""
            if out.ponder_cost is not None:
                extra += f" pc:{out.ponder_cost.item():.2f}"
            if out.mtp_loss is not None:
                extra += f" mtp:{out.mtp_loss.item():.3f}"
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f}{} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{} [FSDP v3-DC]'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, batch_ppl, lr, tps, extra,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

        # 定期保存
        if is_boundary and accum_step % args.save_interval == 0:
            model.eval()
            cur_loss = loss.item() * args.accumulation_steps
            save_checkpoint_fsdp(args.save_dir, model, optimizer,
                                 accum_step, epoch, cur_loss, tokens_seen, rank, config)
            model.train()

    return tokens_seen, accum_step


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuronSpark v3 Deep Copy Pretraining (FSDP)")

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=40)
    parser.add_argument('--v_th_min', type=float, default=0.1)

    # MoE 参数
    parser.add_argument('--num_experts', type=int, default=32)
    parser.add_argument('--moe_top_k', type=int, default=4)
    parser.add_argument('--expert_hidden', type=int, default=1024)
    parser.add_argument('--shared_expert_hidden', type=int, default=2048)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--routed_scaling_factor', type=float, default=2.5)

    # Attention 参数
    parser.add_argument('--n_q_heads', type=int, default=8)
    parser.add_argument('--n_kv_heads', type=int, default=2)
    parser.add_argument('--head_dim', type=int, default=128)

    # MTP 参数
    parser.add_argument('--n_mtp_heads', type=int, default=1)

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints_v3")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=4.5e-4)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # Loss 权重
    parser.add_argument('--ponder_weight', type=float, default=0.01)
    parser.add_argument('--ek_floor', type=float, default=4.0)
    parser.add_argument('--ek_floor_weight', type=float, default=0.1)
    parser.add_argument('--mtp_weight', type=float, default=0.1)

    # FSDP 参数
    parser.add_argument('--sharding_strategy', type=str, default='full_shard',
                        choices=list(SHARDING_STRATEGIES.keys()))

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)

    # 数据
    parser.add_argument("--data_path", type=str,
                        default="data/seq-monkey/seq_monkey_datawhale.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # ==================== 分布式初始化 ====================
    local_rank, rank, world_size = setup_distributed()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(42 + rank)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # ==================== 模型初始化 ====================
    config = build_config(args)
    model, tokenizer = init_model(config, local_rank, rank)

    # 参数分组
    _pg = model.get_param_groups()
    _neuron_keys = {'input_neurons', 'ssm_bias'}
    _no_decay_keys = {'rms_norms', 'halt_projs', 'embedding',
                       'moe_router', 'moe_norms', 'attn_norms'}
    neuron_params = [p for k in _neuron_keys for p in _pg.get(k, [])]
    no_decay_params = [p for k in _no_decay_keys for p in _pg.get(k, [])]
    decay_params = [p for k, ps in _pg.items()
                    if k not in _neuron_keys and k not in _no_decay_keys
                    for p in ps]

    raw_model = model
    model, device = wrap_model_fsdp(model, args, local_rank)

    # ==================== 数据加载 ====================
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        pin_memory=True, drop_last=True, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # ==================== 优化器 (AdamW, Nemotron recipe) ====================
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': args.learning_rate, 'lr_mult': 1.0,
         'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'lr': args.learning_rate, 'lr_mult': 1.0,
         'weight_decay': 0.0},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult), 'weight_decay': 0.0},
    ], betas=(0.9, 0.95))

    # 恢复
    tokens_seen = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch, best_loss, tokens_seen = load_checkpoint_fsdp(
            args.resume, model, optimizer, device, rank,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps * world_size

    # 统计层类型分布
    block_types = config.layers_block_type
    n_ssm = sum(1 for t in block_types if t == 'ssm')
    n_moe = sum(1 for t in block_types if t == 'moe')
    n_attn = sum(1 for t in block_types if t == 'attention')

    Logger(f"\n{'='*60}", rank)
    Logger(f"NeuronSpark v3 Deep Copy Pretraining (FSDP, {world_size} GPUs)", rank)
    Logger(f"  Architecture: {n_ssm} SSM + {n_moe} MoE + {n_attn} Attn = "
           f"{config.num_hidden_layers} layers", rank)
    Logger(f"  Model:       D={config.hidden_size}, N={config.ssm_N}, K={config.ssm_K}", rank)
    Logger(f"  MoE:         {config.n_routed_experts} experts, top-{config.num_experts_per_tok}, "
           f"expert_hidden={config.moe_intermediate_size}, "
           f"scaling={config.routed_scaling_factor}", rank)
    Logger(f"  Attention:   {config.num_attention_heads}Q/{config.num_key_value_heads}KV, "
           f"head_dim={config.head_dim}", rank)
    Logger(f"  MTP:         {config.n_mtp_heads} heads (weight={args.mtp_weight})", rank)
    Logger(f"  Data:        {args.data_path}", rank)
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} × accum "
           f"{args.accumulation_steps} = {effective_batch}", rank)
    Logger(f"  LR:          {args.learning_rate} (WSD: warmup {args.warmup_iters} → stable → "
           f"sqrt decay)", rank)
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} "
           f"({args.neuron_lr_mult}×)", rank)
    Logger(f"  Loss weights: ponder={args.ponder_weight}, ek_floor={args.ek_floor_weight}, "
           f"mtp={args.mtp_weight}", rank)
    Logger(f"{'='*60}\n", rank)

    # ==================== 训练 ====================
    accum_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tokens_seen, accum_step = train_epoch(
            epoch, model, raw_model, train_loader, sampler, optimizer, ctx, args,
            iter_per_epoch, tokens_seen, rank, world_size, config, accum_step=accum_step,
        )

    Logger(f"\nTraining finished. Total tokens seen: {tokens_seen:,}", rank)
    if is_main_process(rank):
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
    save_checkpoint_fsdp(args.save_dir, model, optimizer,
                         args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen,
                         rank, config)

    cleanup_distributed()
