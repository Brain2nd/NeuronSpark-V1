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
from neuronspark import modeling_neuronspark as mns
from utils.param_groups import promote_neuron_params_fp32


def _disable_modulation_checkpoint():
    """Disable the _fused_modulation checkpoint optimization for A/B comparison."""
    import neuronspark.modeling_neuronspark as mns
    mns._FUSED_MODULATION_CHECKPOINT = False


def _install_v1_shims():
    """Monkey-patch plif_*_forward_v2 to use V1 kernels, for A/B comparison.

    V1 returns (output, V_post, None) / (output, V_post). We need v2 signature
    (output, v_last) — but since v_last semantically is V_post[-1], we just
    return (output, V_post[-1]) where V_post[-1] is a VIEW (the original V1
    behavior of holding the storage across steps).
    """
    _orig_parallel = mns.plif_parallel_forward
    _orig_rowparam = mns.plif_rowparam_forward

    def _v1_parallel(beta, u, v_th, v_init):
        output, V_post, _ = _orig_parallel(beta, u, v_th, v_init)
        return output, V_post[-1]    # VIEW — pins V_post storage via .v assignment

    def _v1_rowparam(beta_row, u, v_th_row, v_init):
        output, V_post = _orig_rowparam(beta_row, u, v_th_row, v_init)
        return output, V_post[-1]    # VIEW

    mns.plif_parallel_forward_v2 = _v1_parallel
    mns.plif_rowparam_forward_v2 = _v1_rowparam


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
    ap.add_argument("--use_v1", action="store_true",
                    help="Monkey-patch plif_*_forward_v2 to use V1 (V_post view) for A/B comparison.")
    ap.add_argument("--no_mod_ckpt", action="store_true",
                    help="Disable _fused_modulation activation checkpointing (for A/B).")
    args = ap.parse_args()

    cfg = dict(
        D=args.D, N=args.N, K=args.K, num_layers=args.layers, D_ff=args.D_ff,
        vocab_size=args.vocab_size, memory_layer_interval=4,
    )
    mode = "V1 (V_post view)" if args.use_v1 else "V2 (v_last clone)"
    print(f"Config: {cfg}")
    print(f"Mode:   {mode}")
    print(f"Batch={args.batch}, Seq={args.seq}, n_steps=2\n")

    if args.use_v1:
        _install_v1_shims()
    if args.no_mod_ckpt:
        _disable_modulation_checkpoint()
        print("[patch] _fused_modulation checkpoint DISABLED")

    peak, nparams = run_model(args.batch, args.seq, cfg, n_steps=2)
    print(f"Peak GPU memory: {peak:.3f} GB (params: {nparams:.3f} B)")
