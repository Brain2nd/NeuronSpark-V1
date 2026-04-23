"""Integrated memory benchmark: full model forward+backward with V1 vs V2 PLIF.

Compares real-world peak memory when PLIF call sites use plif_*_forward_v2
(returns v_last, frees V_post) vs plif_*_forward (returns V_post held by .v view).

Runs TWO training steps to expose cross-step memory holding behavior.

Run:
  python scripts/bench_plif_v2_integrated.py
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from utils.param_groups import promote_neuron_params_fp32


def run_model(batch: int, seq: int, cfg_kwargs: dict, n_steps: int = 2):
    torch.manual_seed(0)
    cfg = NeuronSparkConfig(**cfg_kwargs)
    m = NeuronSparkForCausalLM(cfg).cuda().to(torch.bfloat16)
    promote_neuron_params_fp32(m)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    for step in range(n_steps):
        x = torch.randint(0, cfg.vocab_size, (batch, seq), device='cuda')
        y = torch.randint(0, cfg.vocab_size, (batch, seq), device='cuda')
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = m.snn(x, y)
            loss = out.last_loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    peak = torch.cuda.max_memory_allocated() / 1e9
    n_params = sum(p.numel() for p in m.parameters()) / 1e9

    del m, opt, out, loss, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return peak, n_params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--D_ff", type=int, default=1536)
    ap.add_argument("--vocab_size", type=int, default=1024)
    args = ap.parse_args()

    cfg = dict(
        D=args.D, N=args.N, K=args.K, num_layers=args.layers, D_ff=args.D_ff,
        vocab_size=args.vocab_size, memory_layer_interval=4,
    )
    print(f"Config: {cfg}")
    print(f"Batch={args.batch}, Seq={args.seq}, n_steps=2\n")

    peak, nparams = run_model(args.batch, args.seq, cfg, n_steps=2)
    print(f"Peak GPU memory: {peak:.3f} GB (params: {nparams:.3f} B)")
