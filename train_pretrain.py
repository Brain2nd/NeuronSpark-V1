"""NeuronSpark v3 pretrain entry (DeepSpeed ZeRO-2, HF-native).

All model / checkpoint I/O goes through the HuggingFace API
(`NeuronSparkForCausalLM.from_pretrained` / `save_pretrained`). The training loop
accesses the inner `model.snn` for SNN-specific features (ponder cost,
gradient compensation, param groups).

Usage:
    deepspeed --num_gpus=8 train_pretrain.py \
        --config_json configs/v3_base.json \
        --data_path data/v3_pretrain_mix/shards/ \
        --tokenizer_path tokenizer_v3/ \
        --out_dir checkpoints_v3/ \
        --deepspeed_config ds_config.json \
        --batch_size 1 --accumulation_steps 64 --learning_rate 2e-4
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

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from nsdata.pretrain_dataset import PretrainDataset
from utils.dashboard import SNNDashboard
from utils.param_groups import build_param_groups, promote_neuron_params_fp32

warnings.filterwarnings("ignore")


def is_main():
    if not dist.is_initialized():
        return int(os.environ.get("RANK", "0")) == 0
    return dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def cosine_lr(step, total_steps, base_lr, warmup):
    min_lr = base_lr / 10.0
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if step >= total_steps:
        return min_lr
    decay = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def ponder_temperature(step: int, total_steps: int, T_init: float, T_final: float,
                        tau_frac: float = 0.1) -> float:
    """Exponential decay T(step) = T_final + (T_init - T_final) * exp(-step / tau).

    tau = tau_frac * total_steps controls how fast T anneals.
    Default tau_frac=0.1 → 10% of training spent annealing T.
    """
    tau = max(tau_frac * total_steps, 1.0)
    return T_final + (T_init - T_final) * math.exp(-step / tau)


def ponder_exploration(step: int, total_steps: int, eps_init: float,
                        eps_final: float = 0.01) -> float:
    """Linear decay of forced-exploration probability over training."""
    if total_steps <= 1:
        return eps_final
    frac = min(step / total_steps, 1.0)
    return eps_init + (eps_final - eps_init) * frac


def load_model(args) -> tuple[NeuronSparkForCausalLM, AutoTokenizer, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    if args.resume or args.pretrained_ckpt:
        src = args.resume or args.pretrained_ckpt
        log(f"Loading model from {src}")
        model = NeuronSparkForCausalLM.from_pretrained(src, dtype=torch.bfloat16, trust_remote_code=True)
    else:
        if args.config_json:
            with open(args.config_json) as f:
                cfg_kwargs = json.load(f)
        else:
            cfg_kwargs = {}
        for k in ("vocab_size", "D", "N", "K", "num_layers", "D_ff", "v_th_min"):
            v = getattr(args, k, None)
            if v is not None:
                cfg_kwargs[k] = v
        cfg_kwargs.setdefault("vocab_size", len(tokenizer))
        config = NeuronSparkConfig(**cfg_kwargs)
        log(f"Building fresh model from config: {json.dumps(cfg_kwargs, indent=2)}")
        model = NeuronSparkForCausalLM(config)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device=device, dtype=torch.bfloat16)
    n_promoted = promote_neuron_params_fp32(model)
    log(f"  Promoted {n_promoted} neuron params to fp32 master weights")

    n_total = sum(p.numel() for p in model.parameters())
    log(f"  Model parameters: {n_total / 1e6:.1f} M")
    return model, tokenizer, device


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
            if step % 10000 == 0 and rank == 0:
                print(f"  [skip→resume] {step}/{start_step}")
            continue

        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        total_steps = args.epochs * iters_per_epoch
        lr = cosine_lr(step, total_steps, args.learning_rate, args.warmup_iters)
        for g in engine.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

        # v3 PonderNet: temperature + exploration schedule (set BEFORE forward)
        cur_T = ponder_temperature(
            step, total_steps, args.ponder_T_init, args.ponder_T_final,
            tau_frac=args.ponder_tau_frac,
        )
        cur_eps = ponder_exploration(
            step, total_steps, args.eps_explore_init, args.eps_explore_final,
        )
        engine.module.snn.set_ponder_temperature(cur_T)
        engine.module.snn.set_ponder_exploration(cur_eps)

        # Access inner SNNLanguageModel for last_loss + ponder_cost
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            snn = engine.module.snn
            out = snn(X, Y)
            per_token_loss = out.last_loss
            mask_flat = loss_mask.view(-1).float()
            denom = mask_flat.sum().clamp(min=1.0)
            ce_loss = (per_token_loss * mask_flat).sum() / denom
            loss = ce_loss
            if out.ponder_cost is not None and args.ponder_weight > 0:
                loss = loss + args.ponder_weight * out.ponder_cost

        engine.backward(loss)
        is_boundary = (step + 1) % args.accumulation_steps == 0
        if is_boundary:
            engine.module.snn.compensate_modulation_gradients()
        engine.step()

        # v3 PonderNet: gradient-free bias balancing (AFTER step)
        if is_boundary:
            engine.module.snn.update_ponder_bias(
                lr_bias=args.bias_balancing_lr,
                ema_decay=args.bias_balancing_ema,
            )

        valid = loss_mask.sum()
        if world > 1:
            dist.all_reduce(valid, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid.item())

        if is_boundary and dashboard is not None and is_main():
            elapsed = time.time() - t_start
            dashboard.log_step(step, {
                "loss": ce_loss.item(),
                "ppl": math.exp(min(ce_loss.item(), 20.0)),
                "lr": lr,
                "tps": tokens_seen / max(elapsed, 1),
                "tokens_seen": tokens_seen,
                "ponder_cost": float(out.ponder_cost) if out.ponder_cost is not None else 0.0,
                "memory_current_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            }, engine.module, log_params=(step % args.log_interval == 0))

        if step % args.log_interval == 0 and is_main():
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            eta_min = elapsed / (step - start_step + 1) * (iters_per_epoch - step - 1) / 60
            print(
                f"ep[{epoch+1}/{args.epochs}] step {step}/{iters_per_epoch} "
                f"loss {ce_loss.item():.3f} ppl {math.exp(min(ce_loss.item(), 20.0)):.1f} "
                f"lr {lr:.2e} tps {tps:.0f} eta {eta_min:.1f}min "
                f"mem {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            )

        if (step + 1) % args.save_interval == 0:
            if is_main():
                engine.module.eval()
                save_dir = os.path.join(args.out_dir, f"ckpt_step{step+1}")
                engine.module.save_pretrained(save_dir, safe_serialization=True)
                torch.save({
                    "step": step + 1,
                    "epoch": epoch,
                    "tokens_seen": tokens_seen,
                }, os.path.join(save_dir, "training_state.pth"))
                log(f"  → saved {save_dir}")
                if dashboard is not None:
                    dashboard.log_save_point(step, engine.module)
                engine.module.train()
            dist.barrier()

    return tokens_seen


def main():
    ap = argparse.ArgumentParser()
    # Model architecture (if building fresh; --config_json also overrides)
    ap.add_argument("--config_json", type=str, default=None,
                    help="JSON file with NeuronSparkConfig kwargs")
    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    ap.add_argument("--D_ff", type=int, default=None)
    ap.add_argument("--v_th_min", type=float, default=None)
    # Data
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--tokenizer_path", default="tokenizer_v3/")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=8)
    # Training
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accumulation_steps", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--neuron_lr_mult", type=float, default=10.0)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ponder_weight", type=float, default=0.0,
                    help="Weight for ponder_cost (E[k_t]) added to CE loss. 0 = pure CE.")
    # v3 PonderNet schedule
    ap.add_argument("--ponder_T_init", type=float, default=2.0,
                    help="Initial Gumbel temperature (high = explore)")
    ap.add_argument("--ponder_T_final", type=float, default=0.3,
                    help="Final Gumbel temperature (low = near hard argmax)")
    ap.add_argument("--ponder_tau_frac", type=float, default=0.1,
                    help="T anneal time constant as fraction of total_steps")
    ap.add_argument("--eps_explore_init", type=float, default=0.05,
                    help="Forced-exploration prob at step 0")
    ap.add_argument("--eps_explore_final", type=float, default=0.01,
                    help="Forced-exploration prob at end of training")
    ap.add_argument("--bias_balancing_lr", type=float, default=1e-3,
                    help="Gradient-free bias update step size")
    ap.add_argument("--bias_balancing_ema", type=float, default=0.99,
                    help="EMA decay for usage statistics")
    # Checkpoint
    ap.add_argument("--out_dir", default="checkpoints_v3/")
    ap.add_argument("--resume", default=None, help="resume (load model + step/tokens)")
    ap.add_argument("--pretrained_ckpt", default=None,
                    help="init weights only, fresh optimizer")
    # Logging
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--dashboard_dir", default=None)
    # DeepSpeed
    ap.add_argument("--local_rank", type=int, default=-1)
    ap = deepspeed.add_config_arguments(ap)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Model + tokenizer
    model, tokenizer, device = load_model(args)

    # Training state
    start_step, tokens_seen = 0, 0
    if args.resume and os.path.isdir(args.resume):
        ts_path = os.path.join(args.resume, "training_state.pth")
        if os.path.isfile(ts_path):
            ts = torch.load(ts_path, map_location=device, weights_only=False)
            start_step = ts.get("step", 0)
            tokens_seen = ts.get("tokens_seen", 0)
            log(f"  Resume: step={start_step}, tokens={tokens_seen:,}")

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
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length, pad_token_id=pad_id)
    rank = dist.get_rank()
    world = dist.get_world_size()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        pin_memory=True, drop_last=True, num_workers=args.num_workers)
    iters_per_epoch = len(loader)
    eff_batch = args.batch_size * args.accumulation_steps * world

    log(f"\n{'='*60}")
    log(f"NeuronSpark v3 Pretrain — DeepSpeed ZeRO-2, {world} GPUs")
    log(f"  Data              : {args.data_path}  ({len(ds):,} samples)")
    log(f"  Tokenizer         : {args.tokenizer_path}  (vocab={len(tokenizer)})")
    log(f"  Sequence length   : {args.max_length}")
    log(f"  Effective batch   : {args.batch_size} × {args.accumulation_steps} × {world} = {eff_batch}")
    log(f"  Steps/epoch       : {iters_per_epoch:,}")
    log(f"  LR                : {args.learning_rate} (warmup {args.warmup_iters} → cosine)")
    log(f"  Neuron LR         : {args.learning_rate * args.neuron_lr_mult}")
    log(f"  Save every        : {args.save_interval} steps")
    log(f"{'='*60}\n")

    # Dashboard
    dashboard = None
    if args.dashboard_dir:
        dashboard = SNNDashboard(args.dashboard_dir, engine.module, rank=rank)

    # Train
    for epoch in range(args.epochs):
        tokens_seen = train_epoch(epoch, engine, loader, sampler, args,
                                   iters_per_epoch, tokens_seen, dashboard, start_step)
        start_step = 0

    if dashboard is not None:
        dashboard.close()
    log("Training complete.")
    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    main()
