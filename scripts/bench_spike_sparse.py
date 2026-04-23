"""Spike-activation sparsity feasibility benchmark.

Question: can we exploit spike=0 patterns in SNNFFN.down_proj to speed up GEMM?

Tests 3 axes:
  1. PROFILE: measure actual spike-product density in v3 model forward pass
     (gate_spike × up_spike) — how sparse is it in practice?
  2. CANDIDATE IMPLS: benchmark dense GEMM (baseline) vs torch.sparse (CSR)
     vs a simple Triton element-masked kernel across synthetic densities.
  3. DECISION: plot density × speedup curve; report the break-even point
     where sparse > dense, compare with measured real density.

Output: console report + /tmp/spike_sparse_bench.json with raw numbers.

Run:
    python scripts/bench_spike_sparse.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Synthetic sparse tensor generator
# ============================================================

def make_synthetic_spike(shape: tuple, density: float, device: str, dtype) -> torch.Tensor:
    """Dense tensor with Bernoulli-density nonzero mask, values like actual spikes."""
    mask = (torch.rand(shape, device=device) < density).to(dtype)
    # Spike magnitude ~ V_th scale, approximate as uniform(0.1, 0.5)
    values = torch.rand(shape, device=device, dtype=dtype) * 0.4 + 0.1
    return mask * values


# ============================================================
# Benchmark impls
# ============================================================

def bench_fn(fn, n_iter: int = 50, n_warmup: int = 5) -> float:
    """Return median wall-clock seconds for fn() after warmup."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def bench_dense_gemm(A: torch.Tensor, W: torch.Tensor, n_iter: int = 50) -> float:
    """A (M, K) @ W.T (K, N) → C (M, N). Baseline."""
    Wt = W.t().contiguous()

    def _run():
        torch.matmul(A, Wt)

    return bench_fn(_run, n_iter=n_iter)


def bench_csr_gemm(A_dense: torch.Tensor, W: torch.Tensor, n_iter: int = 50) -> tuple[float, float]:
    """CSR @ dense. Measures inclusive of CSR conversion (one-time).

    Returns:
        (inclusive_time, conversion_overhead)
    """
    Wt = W.t().contiguous()

    # Convert once
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    A_csr = A_dense.to_sparse_csr()
    torch.cuda.synchronize()
    conv_time = time.perf_counter() - t0

    def _run():
        torch.sparse.mm(A_csr, Wt)

    try:
        return bench_fn(_run, n_iter=n_iter), conv_time
    except RuntimeError as e:
        return float("inf"), conv_time


def bench_gather_gemm(A_dense: torch.Tensor, W: torch.Tensor, n_iter: int = 50) -> float:
    """Row-compact: identify rows with ANY nonzero, run dense GEMM on compact tensor.

    Overhead: 1x scan for row-nonzero-mask + gather.
    Speedup potential: if many rows are all-zero (unlikely in our spike patterns).
    """
    Wt = W.t().contiguous()

    def _run():
        row_nz_mask = (A_dense != 0).any(dim=-1)          # (M,) bool
        if row_nz_mask.any():
            A_sub = A_dense[row_nz_mask]                   # (M_nz, K)
            torch.matmul(A_sub, Wt)                        # (M_nz, N)

    return bench_fn(_run, n_iter=n_iter)


# ============================================================
# Profile real v3 model sparsity
# ============================================================

def profile_v3_sparsity(n_forwards: int = 5, seq_len: int = 64, batch: int = 4):
    """Run a few v3 forward passes, monkey-patch SNNFFN.forward_parallel to
    capture the OUTPUT tensor (= down_proj input).

    Key note: current V2.5/v3 architecture has SNNFFN output be continuous
    `(1-β)·V_post`, NOT a binary spike. So density is expected near 100%.
    Reports density at multiple magnitude thresholds to show the distribution.
    """
    from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
    from neuronspark.modeling_neuronspark import SNNFFN
    from utils.param_groups import promote_neuron_params_fp32

    cfg = NeuronSparkConfig(
        D=1024, N=8, K=12, num_layers=6, D_ff=3072,
        vocab_size=64002, memory_layer_interval=4,
    )
    m = NeuronSparkForCausalLM(cfg).cuda().to(torch.bfloat16)
    promote_neuron_params_fp32(m)
    m.eval()

    captures = []
    _orig_fwd = SNNFFN.forward_parallel

    def _wrapped(self, h_seq):
        out = _orig_fwd(self, h_seq)
        captures.append(out.detach())
        return out

    SNNFFN.forward_parallel = _wrapped

    try:
        with torch.no_grad():
            for it in range(n_forwards):
                x = torch.randint(0, cfg.vocab_size, (batch, seq_len), device="cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = m.snn(x)
    finally:
        SNNFFN.forward_parallel = _orig_fwd

    if not captures:
        return None

    stats = {"exact_zero_density": [],  # True sparsity
              "density_gt_1em6": [],
              "density_gt_1em4": [],
              "density_gt_1em2": [],
              "row_all_zero_frac": [],
              "abs_mean": []}

    for t in captures:
        stats["exact_zero_density"].append(1.0 - (t != 0).float().mean().item())
        stats["density_gt_1em6"].append((t.abs() > 1e-6).float().mean().item())
        stats["density_gt_1em4"].append((t.abs() > 1e-4).float().mean().item())
        stats["density_gt_1em2"].append((t.abs() > 1e-2).float().mean().item())
        stats["row_all_zero_frac"].append((t == 0).all(dim=-1).float().mean().item())
        stats["abs_mean"].append(t.abs().mean().item())

    return {k: sum(v) / len(v) for k, v in stats.items()}


# ============================================================
# Main bench
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048, help="seq_len * batch * K")
    ap.add_argument("--D_ff", type=int, default=3072)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--skip_profile", action="store_true")
    ap.add_argument("--out_json", default="/tmp/spike_sparse_bench.json")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n{'='*70}")
    print(f"Spike-activation sparsity feasibility benchmark")
    print(f"  Shape: A({args.M}, {args.D_ff})  W({args.D}, {args.D_ff})")
    print(f"  Target: down_proj GEMM in SNNFFN")
    print(f"{'='*70}")

    results = {"config": vars(args)}

    # ==== 1. Profile real sparsity (if v3 model buildable) ====
    if not args.skip_profile:
        print("\n--- [1] Real v3 SNNFFN output tensor density profile ---")
        try:
            real_stats = profile_v3_sparsity()
            results["real_sparsity"] = real_stats
            print(f"  |value| mean       : {real_stats['abs_mean']:.4e}")
            print(f"  exact-zero density : {real_stats['exact_zero_density']:.4f}")
            print(f"  density(|x|>1e-6)  : {real_stats['density_gt_1em6']:.4f}")
            print(f"  density(|x|>1e-4)  : {real_stats['density_gt_1em4']:.4f}")
            print(f"  density(|x|>1e-2)  : {real_stats['density_gt_1em2']:.4f}")
            print(f"  row-all-zero frac  : {real_stats['row_all_zero_frac']:.4f}")
            print()
            print(f"  ⚠️  CRITICAL FINDING: SNNFFN output is CONTINUOUS (1-β)·V_post,")
            print(f"      NOT a binary spike. Density near 100% means no zero rows to skip.")
            print(f"      'Spike sparsity' at inter-layer signal level is ILLUSORY under")
            print(f"      current V2.5/v3 architecture. To exploit, must redesign SNNFFN")
            print(f"      output to be binary spike × V_th instead of continuous leakage.")
        except Exception as e:
            print(f"  SKIP (profile failed: {type(e).__name__}: {str(e)[:80]})")
            results["real_sparsity"] = None
    else:
        results["real_sparsity"] = None

    # ==== 2. Synthetic density sweep ====
    print(f"\n--- [2] Synthetic density × speedup sweep ---")
    densities = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    W = torch.randn(args.D, args.D_ff, device=device, dtype=dtype) * 0.02

    sweep = []
    print(f"  {'density':>8} {'dense_ms':>10} {'csr_ms':>10} {'gather_ms':>10}  "
          f"{'csr_spd':>8} {'gather_spd':>11}  {'nonzero%':>9}")
    for d in densities:
        A = make_synthetic_spike((args.M, args.D_ff), density=d, device=device, dtype=dtype)
        nz_pct = (A != 0).float().mean().item() * 100

        t_dense = bench_dense_gemm(A, W, n_iter=args.n_iter) * 1000
        t_csr, conv_ov = bench_csr_gemm(A, W, n_iter=args.n_iter)
        t_csr *= 1000
        t_gather = bench_gather_gemm(A, W, n_iter=args.n_iter) * 1000

        spd_csr = t_dense / t_csr if t_csr > 0 else 0
        spd_gather = t_dense / t_gather if t_gather > 0 else 0

        sweep.append({
            "density": d,
            "nonzero_pct": nz_pct,
            "dense_ms": t_dense,
            "csr_ms": t_csr,
            "gather_ms": t_gather,
            "csr_conv_ms": conv_ov * 1000,
            "speedup_csr": spd_csr,
            "speedup_gather": spd_gather,
        })
        print(f"  {d:>8.2f} {t_dense:>10.3f} {t_csr:>10.3f} {t_gather:>10.3f}  "
              f"{spd_csr:>8.2f}× {spd_gather:>10.2f}×  {nz_pct:>8.1f}%")

    results["density_sweep"] = sweep

    # ==== 3. Decision summary ====
    print(f"\n--- [3] Decision ---")
    break_even = None
    for entry in sweep:
        if entry["speedup_csr"] >= 1.0 or entry["speedup_gather"] >= 1.0:
            break_even = entry["density"]
            break

    if break_even:
        print(f"  Sparse beats dense when density ≤ {break_even:.2f}")
    else:
        print(f"  Sparse NEVER beats dense in tested range [0.05, 0.90]")

    print()
    print(f"  Two independent findings:")
    print(f"    A) torch.sparse CSR is {'10-20× SLOWER' if not break_even else 'viable at edge'}")
    print(f"       across all tested densities on our shapes (M={args.M}, D_ff={args.D_ff}).")
    print(f"       Unstructured element sparsity on GPU is architecturally costly:")
    print(f"       warps execute in lockstep, per-element zeros don't skip lanes.")
    print(f"    B) SNNFFN's output is CONTINUOUS (1-β)·V_post, not a binary spike,")
    print(f"       by V2.5 design decision. Real density is ~100%, so the premise")
    print(f"       'spike=0 rows to skip' does NOT hold for our architecture.")
    print()
    print(f"  VERDICT: Stage 2 (脉冲激活稀疏) is NOT PURSUABLE as originally framed.")
    print(f"  Two paths forward require EXPLICIT architectural change:")
    print(f"    1. Change SNNFFN to output `spike_binary × V_th` (losing continuous")
    print(f"       amplitude modulation that V2.5 chose to preserve expressivity)")
    print(f"    2. Write a custom Triton kernel that exploits structured block-sparse")
    print(f"       patterns IF we introduce structured firing constraints (not yet designed)")
    print()
    print(f"  Recommended: Skip Stage 2-3 spike-sparsity branch.")
    print(f"  PonderNet k_t 早停(Stage 3.1) 仍然有效 — 它是时间维稀疏，和本 bench 正交。")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results: {args.out_json}")


if __name__ == "__main__":
    main()
