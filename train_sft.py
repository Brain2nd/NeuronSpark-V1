"""NeuronSpark v3 SFT entry (DeepSpeed ZeRO-2, HF-native, ChatML).

Loads pretrained checkpoint via `NeuronSparkForCausalLM.from_pretrained`, applies
ChatML mask so loss is computed only on assistant tokens.

Usage:
    deepspeed --num_gpus=8 train_sft.py \
        --pretrained_ckpt checkpoints_v3/ckpt_stepN/ \
        --data_path data/v3_sft_mix/ \
        --tokenizer_path tokenizer_v3/ \
        --out_dir checkpoints_v3_sft/ \
        --deepspeed_config ds_config.json \
        --batch_size 2 --accumulation_steps 32 --learning_rate 5e-5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from neuronspark import NeuronSparkForCausalLM
from nsdata.sft_dataset import SFTDataset
from utils.dashboard import SNNDashboard
from utils.param_groups import build_param_groups, promote_neuron_params_fp32

warnings.filterwarnings("ignore")


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def cosine_lr(step, total, base_lr, warmup):
    min_lr = base_lr / 10.0
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    decay = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def train_epoch(epoch, engine, loader, sampler, args, iters_per_epoch,
                tokens_seen, dashboard, start_step):
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    sampler.set_epoch(epoch)
    t_start = time.time()

    for step, (X, Y, loss_mask) in enumerate(loader):
        if step < start_step:
            continue
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        lr = cosine_lr(step, args.epochs * iters_per_epoch, args.learning_rate, args.warmup_iters)
        for g in engine.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            snn = engine.module.snn
            out = snn(X, Y)
            per_token = out.last_loss
            mask_flat = loss_mask.view(-1).float()
            denom = mask_flat.sum().clamp(min=1.0)
            ce_loss = (per_token * mask_flat).sum() / denom
            loss = ce_loss
            if out.ponder_cost is not None and args.ponder_weight > 0:
                loss = loss + args.ponder_weight * out.ponder_cost

        engine.backward(loss)
        is_boundary = (step + 1) % args.accumulation_steps == 0
        if is_boundary:
            engine.module.snn.compensate_modulation_gradients()
            if dashboard is not None:  # ALL ranks call (collective all_reduce inside)
                dashboard.cache_grad_norms(engine.module)
        engine.step()

        valid = loss_mask.sum()
        if world > 1:
            dist.all_reduce(valid, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid.item())

        if step % args.log_interval == 0 and is_main():
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            print(
                f"ep[{epoch+1}/{args.epochs}] step {step}/{iters_per_epoch} "
                f"loss {ce_loss.item():.3f} ppl {math.exp(min(ce_loss.item(), 20.0)):.1f} "
                f"lr {lr:.2e} tps {tps:.0f} mem {torch.cuda.memory_allocated()/1e9:.1f}GB"
            )
        if is_boundary and dashboard is not None and is_main():
            elapsed = time.time() - t_start
            dashboard.log_step(step, {
                "loss": ce_loss.item(),
                "ppl": math.exp(min(ce_loss.item(), 20.0)),
                "lr": lr,
                "tps": tokens_seen / max(elapsed, 1),
                "tokens_seen": tokens_seen,
                "ponder_cost": float(out.ponder_cost) if out.ponder_cost is not None else 0.0,
            }, engine.module, log_params=(step % args.log_interval == 0))

        if (step + 1) % args.save_interval == 0:
            if is_main():
                save_dir = os.path.join(args.out_dir, f"ckpt_step{step+1}")
                engine.module.save_pretrained(save_dir, safe_serialization=True)
                torch.save({"step": step + 1, "epoch": epoch, "tokens_seen": tokens_seen},
                           os.path.join(save_dir, "training_state.pth"))
                log(f"  → saved {save_dir}")
                if dashboard is not None:
                    dashboard.log_save_point(step, engine.module)
            dist.barrier()

    return tokens_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True,
                    help="HF-format NeuronSpark checkpoint to initialize from")
    ap.add_argument("--data_path", required=True, help="HF Arrow directory or JSONL")
    ap.add_argument("--tokenizer_path", default="tokenizer_v3/")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accumulation_steps", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--neuron_lr_mult", type=float, default=5.0)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ponder_weight", type=float, default=0.0)
    ap.add_argument("--out_dir", default="checkpoints_v3_sft/")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--dashboard_dir", default=None)
    ap.add_argument("--local_rank", type=int, default=-1)
    ap = deepspeed.add_config_arguments(ap)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer (SFT: eos = <|im_end|> for early-stopping at turn boundary)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    log(f"Tokenizer {args.tokenizer_path}: vocab={len(tokenizer)}, im_end_id={im_end_id}")

    # Load pretrained model
    log(f"Loading pretrained from {args.pretrained_ckpt}")
    model = NeuronSparkForCausalLM.from_pretrained(args.pretrained_ckpt, dtype=torch.bfloat16, trust_remote_code=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device=device, dtype=torch.bfloat16)
    n_fp32 = promote_neuron_params_fp32(model)
    log(f"  Promoted {n_fp32} neuron params to fp32")

    # Optimizer
    param_groups = build_param_groups(
        model, learning_rate=args.learning_rate, neuron_lr_mult=args.neuron_lr_mult,
    )
    optimizer = torch.optim.Adam(param_groups)

    # DeepSpeed
    ds_cfg_path = getattr(args, "deepspeed_config", "ds_config.json")
    with open(ds_cfg_path) as f:
        ds_cfg = json.load(f)
    ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_cfg["gradient_accumulation_steps"] = args.accumulation_steps
    ds_cfg["gradient_clipping"] = args.grad_clip

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_cfg,
    )

    # Data
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_length, pad_token_id=pad_id)
    rank = dist.get_rank()
    world = dist.get_world_size()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        pin_memory=True, drop_last=True, num_workers=args.num_workers)

    iters_per_epoch = len(loader)
    eff_batch = args.batch_size * args.accumulation_steps * world

    log(f"\n{'='*60}")
    log(f"NeuronSpark v3 SFT — DeepSpeed ZeRO-2, {world} GPUs")
    log(f"  Pretrained        : {args.pretrained_ckpt}")
    log(f"  Data              : {args.data_path}  ({len(ds):,} samples)")
    log(f"  Effective batch   : {eff_batch}, Steps/epoch: {iters_per_epoch:,}")
    log(f"  LR                : {args.learning_rate} (warmup {args.warmup_iters})")
    log(f"  Neuron LR         : {args.learning_rate * args.neuron_lr_mult}")
    log(f"{'='*60}\n")

    dashboard = SNNDashboard(args.dashboard_dir, engine.module, rank=rank) if args.dashboard_dir else None
    tokens_seen = 0
    for epoch in range(args.epochs):
        tokens_seen = train_epoch(
            epoch, engine, loader, sampler, args, iters_per_epoch,
            tokens_seen, dashboard, start_step=0,
        )
    if dashboard is not None:
        dashboard.close()
    log("SFT complete.")
    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    main()
