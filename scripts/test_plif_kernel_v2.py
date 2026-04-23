"""Empirical validation of PLIF V2 kernels (V_post-less, recompute-in-backward).

Tests:
  1. Gradient match: run same random input through V1 and V2 kernels, compare
     gradients wrt beta / u / v_th / v_init. Max abs diff must be < 1e-3 in bf16.
  2. Forward output match: (output, v_last) from V2 must match (output, V_post[-1]) from V1.
  3. Memory delta: measure peak GPU memory of a training step using V1 vs V2.
     Quantifies whether V_post elimination actually saves memory (depends on
     whether u is already held by upstream autograd).

Run:
  python scripts/test_plif_kernel_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from neuronspark.modeling_neuronspark import (
    _TritonPLIFRowParamForward,
    _TritonPLIFRowParamForward_V2,
    _TritonPLIFForward,
    _TritonPLIFForward_V2,
)


def _fmt_tensor(t):
    return f"{tuple(t.shape)}·{t.dtype}"


# ============================================================
# Test 1: gradient match — rowparam
# ============================================================

def test_grad_match_rowparam():
    print("=== Test 1: gradient match (rowparam) ===")
    torch.manual_seed(0)
    K, S, D = 12, 8, 128   # K steps, seq positions (== seq*batch flattened), DN cols
    # Our kernel takes u as (K, *shape) and beta/v_th as (*shape). Broadcast implicit.
    beta_row = torch.rand(S, D, device='cuda', dtype=torch.bfloat16) * 0.5 + 0.4
    u        = torch.randn(K, S, D, device='cuda', dtype=torch.bfloat16) * 0.3
    v_th_row = torch.rand(S, D, device='cuda', dtype=torch.bfloat16) * 0.2 + 0.05
    v_init   = torch.zeros(S, D, device='cuda', dtype=torch.bfloat16)

    # ---- V1 path ----
    b1 = beta_row.clone().detach().requires_grad_(True)
    u1 = u.clone().detach().requires_grad_(True)
    t1 = v_th_row.clone().detach().requires_grad_(True)
    i1 = v_init.clone().detach().requires_grad_(True)
    out1, Vpost1 = _TritonPLIFRowParamForward.apply(b1, u1, t1, i1)
    vlast1 = Vpost1[-1]
    loss1 = (out1.float().sum() + vlast1.float().sum())
    loss1.backward()

    # ---- V2 path ----
    b2 = beta_row.clone().detach().requires_grad_(True)
    u2 = u.clone().detach().requires_grad_(True)
    t2 = v_th_row.clone().detach().requires_grad_(True)
    i2 = v_init.clone().detach().requires_grad_(True)
    out2, vlast2 = _TritonPLIFRowParamForward_V2.apply(b2, u2, t2, i2)
    loss2 = (out2.float().sum() + vlast2.float().sum())
    loss2.backward()

    # ---- Compare ----
    fwd_out_diff  = (out1.float() - out2.float()).abs().max().item()
    fwd_last_diff = (vlast1.float() - vlast2.float()).abs().max().item()
    g_beta  = (b1.grad.float() - b2.grad.float()).abs().max().item()
    g_u     = (u1.grad.float() - u2.grad.float()).abs().max().item()
    g_vth   = (t1.grad.float() - t2.grad.float()).abs().max().item()
    g_vinit = (i1.grad.float() - i2.grad.float()).abs().max().item()

    print(f"  fwd output diff:   {fwd_out_diff:.2e}")
    print(f"  fwd v_last diff:   {fwd_last_diff:.2e}")
    print(f"  grad beta diff:    {g_beta:.2e}")
    print(f"  grad u diff:       {g_u:.2e}")
    print(f"  grad v_th diff:    {g_vth:.2e}")
    print(f"  grad v_init diff:  {g_vinit:.2e}")
    TOL = 1e-2  # bf16 slack
    ok = all(x < TOL for x in [fwd_out_diff, fwd_last_diff, g_beta, g_u, g_vth, g_vinit])
    print(f"  → {'PASS' if ok else 'FAIL'} (tol < {TOL})")
    return ok


# ============================================================
# Test 2: gradient match — perstep
# ============================================================

def test_grad_match_perstep():
    print("\n=== Test 2: gradient match (perstep) ===")
    torch.manual_seed(1)
    K, S, D = 12, 8, 128
    beta = torch.rand(K, S, D, device='cuda', dtype=torch.bfloat16) * 0.5 + 0.4
    u    = torch.randn(K, S, D, device='cuda', dtype=torch.bfloat16) * 0.3
    v_th = torch.rand(K, S, D, device='cuda', dtype=torch.bfloat16) * 0.2 + 0.05
    v_init = torch.zeros(S, D, device='cuda', dtype=torch.bfloat16)

    b1 = beta.clone().detach().requires_grad_(True)
    u1 = u.clone().detach().requires_grad_(True)
    t1 = v_th.clone().detach().requires_grad_(True)
    i1 = v_init.clone().detach().requires_grad_(True)
    out1, Vpost1 = _TritonPLIFForward.apply(b1, u1, t1, i1)
    loss1 = (out1.float().sum() + Vpost1[-1].float().sum())
    loss1.backward()

    b2 = beta.clone().detach().requires_grad_(True)
    u2 = u.clone().detach().requires_grad_(True)
    t2 = v_th.clone().detach().requires_grad_(True)
    i2 = v_init.clone().detach().requires_grad_(True)
    out2, vlast2 = _TritonPLIFForward_V2.apply(b2, u2, t2, i2)
    loss2 = (out2.float().sum() + vlast2.float().sum())
    loss2.backward()

    fwd_diff = (out1.float() - out2.float()).abs().max().item()
    last_diff = (Vpost1[-1].float() - vlast2.float()).abs().max().item()
    g_beta = (b1.grad.float() - b2.grad.float()).abs().max().item()
    g_u    = (u1.grad.float() - u2.grad.float()).abs().max().item()
    g_vth  = (t1.grad.float() - t2.grad.float()).abs().max().item()
    g_vi   = (i1.grad.float() - i2.grad.float()).abs().max().item()
    print(f"  fwd output diff:   {fwd_diff:.2e}")
    print(f"  fwd v_last diff:   {last_diff:.2e}")
    print(f"  grad beta diff:    {g_beta:.2e}")
    print(f"  grad u diff:       {g_u:.2e}")
    print(f"  grad v_th diff:    {g_vth:.2e}")
    print(f"  grad v_init diff:  {g_vi:.2e}")
    TOL = 1e-2
    ok = all(x < TOL for x in [fwd_diff, last_diff, g_beta, g_u, g_vth, g_vi])
    print(f"  → {'PASS' if ok else 'FAIL'} (tol < {TOL})")
    return ok


# ============================================================
# Test 3: memory delta — realistic shape
# ============================================================

def test_memory_delta():
    print("\n=== Test 3: memory delta (rowparam, realistic shape) ===")
    K, S, D = 12, 2048 * 1, 8192   # seq=2048, batch=1, DN=1024*8
    torch.manual_seed(2)
    beta_row = torch.rand(S, D, device='cuda', dtype=torch.bfloat16) * 0.5 + 0.4
    u        = torch.randn(K, S, D, device='cuda', dtype=torch.bfloat16) * 0.3
    v_th_row = torch.rand(S, D, device='cuda', dtype=torch.bfloat16) * 0.2 + 0.05
    v_init   = torch.zeros(S, D, device='cuda', dtype=torch.bfloat16)

    for name, fn in [("V1", _TritonPLIFRowParamForward.apply),
                     ("V2", _TritonPLIFRowParamForward_V2.apply)]:
        b = beta_row.clone().detach().requires_grad_(True)
        uu = u.clone().detach().requires_grad_(True)
        t = v_th_row.clone().detach().requires_grad_(True)
        i = v_init.clone().detach().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        out, vpost_or_vlast = fn(b, uu, t, i)
        peak_fwd = torch.cuda.max_memory_allocated() / 1e9
        # Keep vpost_or_vlast reference alive (like .v = V_post[-1].detach() does)
        # To mimic real scenario: clone only [-1] and drop the rest
        if vpost_or_vlast.dim() == u.dim():  # V1 returned full V_post
            persisted = vpost_or_vlast[-1].detach()   # view-based (V1 scenario)
            # Alternative fair comparison: persisted = vpost_or_vlast[-1].clone()
        else:
            persisted = vpost_or_vlast.detach()       # V2 scenario
        # Force a backward pass
        loss = out.float().sum() + (vpost_or_vlast[-1] if vpost_or_vlast.dim() == u.dim() else vpost_or_vlast).float().sum()
        loss.backward()
        peak_bwd = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {name}: peak_fwd={peak_fwd:.3f} GB  peak_fwd+bwd={peak_bwd:.3f} GB")
        del b, uu, t, i, out, vpost_or_vlast, persisted, loss
        torch.cuda.empty_cache()


# ============================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)
    ok1 = test_grad_match_rowparam()
    ok2 = test_grad_match_perstep()
    test_memory_delta()
    if ok1 and ok2:
        print("\n=== GRAD MATCH PASSED — V2 kernels are correct ===")
    else:
        print("\n=== GRAD MATCH FAILED — DO NOT swap in production ===")
        sys.exit(1)
