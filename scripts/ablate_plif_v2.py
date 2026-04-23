"""V2 PLIF 反向路径消融：isolate每个组件的真实耗时。

目标：回答"V2 反向比 V1 慢 18% 的 157 ms 到底花在哪"。

步骤：
  A) V1 反向：基线。只跑 bwd kernel。
  B) V2 只分配 scratch tensors，不跑 kernel。
  C) V2 跑 forward-recompute kernel（Phase 1）。
  D) V2 跑 backward kernel（Phase 2，和 V1 一样）。
  E) V2 完整：A+C+D 等价。

测真实时间（50 次 warmup + 100 次 timing，CUDA events）。

Run:
  python scripts/ablate_plif_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuronspark.modeling_neuronspark import (
    _fused_plif_fwd_rowparam_kernel,
    _fused_plif_bwd_rowparam_kernel,
    _TritonPLIFRowParamForward,
    _TritonPLIFRowParamForward_V2,
)

# Realistic shape from bench: seq=2048, batch=1, DN=8192 (D=512, N=16), K=12
# 用更大 DN（接近生产）
K, S, DN = 12, 2048, 8192
BLOCK = 128


def setup():
    torch.manual_seed(0)
    beta_row = torch.rand(S, DN, device='cuda', dtype=torch.bfloat16) * 0.5 + 0.4
    u = torch.randn(K, S, DN, device='cuda', dtype=torch.bfloat16) * 0.3
    v_th_row = torch.rand(S, DN, device='cuda', dtype=torch.bfloat16) * 0.2 + 0.05
    v_init = torch.zeros(S, DN, device='cuda', dtype=torch.bfloat16)
    return beta_row, u, v_th_row, v_init


def time_it(fn, warmup=20, iters=50):
    """返回 (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts, ends = [], []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        starts.append(s); ends.append(e)
    torch.cuda.synchronize()
    import statistics
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.mean(times), statistics.stdev(times)


# =====================================================
# Scenarios
# =====================================================

def run_v1_forward():
    b, u, t, i = setup()
    b = b.clone().requires_grad_()
    u = u.clone().requires_grad_()
    t = t.clone().requires_grad_()
    i = i.clone().requires_grad_()
    out, vpost = _TritonPLIFRowParamForward.apply(b, u, t, i)
    return out, vpost, (b, u, t, i)


def run_v2_forward():
    b, u, t, i = setup()
    b = b.clone().requires_grad_()
    u = u.clone().requires_grad_()
    t = t.clone().requires_grad_()
    i = i.clone().requires_grad_()
    out, vlast = _TritonPLIFRowParamForward_V2.apply(b, u, t, i)
    return out, vlast, (b, u, t, i)


def v1_backward_only(out, vpost):
    """只跑 V1 backward（用 saved V_post）"""
    loss = (out.float().sum() + vpost.float().sum())
    loss.backward(retain_graph=True)


def v2_backward_only(out, vlast):
    loss = (out.float().sum() + vlast.float().sum())
    loss.backward(retain_graph=True)


def only_alloc():
    """只做 V2 backward 的 tensor 分配，不跑 kernel."""
    b, u, t, i = setup()
    V_post = torch.empty_like(u)
    output_scratch = torch.empty_like(u)
    grad_V_post = torch.zeros_like(V_post)
    grad_beta_row = torch.empty_like(b)
    grad_u = torch.empty_like(u)
    grad_v_th_row = torch.empty_like(t)
    grad_v_init = torch.empty_like(i)
    return V_post, output_scratch, grad_V_post, grad_beta_row, grad_u, grad_v_th_row, grad_v_init


def only_fwd_kernel():
    """只跑 forward kernel 1 次（V2 Phase 1 重建 V_post）."""
    b, u, t, i = setup()
    output = torch.empty_like(u)
    V_post = torch.empty_like(u)
    num_cols = u[0].numel()
    grid = ((num_cols + BLOCK - 1) // BLOCK,)
    _fused_plif_fwd_rowparam_kernel[grid](
        b.contiguous(), u.contiguous(), t.contiguous(), i.contiguous(),
        output, V_post, K, num_cols, BLOCK=BLOCK,
    )


def only_bwd_kernel():
    """只跑 backward kernel 1 次（V1 完整反向 = V2 Phase 2）."""
    b, u, t, i = setup()
    output = torch.empty_like(u)
    V_post = torch.empty_like(u)
    num_cols = u[0].numel()
    grid = ((num_cols + BLOCK - 1) // BLOCK,)
    _fused_plif_fwd_rowparam_kernel[grid](
        b.contiguous(), u.contiguous(), t.contiguous(), i.contiguous(),
        output, V_post, K, num_cols, BLOCK=BLOCK,
    )
    grad_output = torch.ones_like(output)
    grad_V_post = torch.zeros_like(V_post)
    grad_beta_row = torch.empty_like(b)
    grad_u = torch.empty_like(u)
    grad_v_th_row = torch.empty_like(t)
    grad_v_init = torch.empty_like(i)
    _fused_plif_bwd_rowparam_kernel[grid](
        b.contiguous(), t.contiguous(), i.contiguous(), V_post, output,
        grad_output, grad_V_post,
        grad_beta_row, grad_u, grad_v_th_row, grad_v_init,
        K, num_cols, BLOCK=BLOCK,
    )


def v1_full():
    """V1 完整反向：只调 bwd kernel，V_post 已 saved."""
    b, u, t, i = setup()
    b.requires_grad_(); u.requires_grad_(); t.requires_grad_(); i.requires_grad_()
    output, V_post = _TritonPLIFRowParamForward.apply(b, u, t, i)
    # backward 会调 _fused_plif_bwd_rowparam_kernel 1 次
    (output.float().sum() + V_post.float().sum()).backward()


def v2_full():
    """V2 完整反向：调 fwd kernel 重建 + bwd kernel. 2 次 kernel."""
    b, u, t, i = setup()
    b.requires_grad_(); u.requires_grad_(); t.requires_grad_(); i.requires_grad_()
    output, vlast = _TritonPLIFRowParamForward_V2.apply(b, u, t, i)
    (output.float().sum() + vlast.float().sum()).backward()


# =====================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)

    print(f"Shape: K={K}, S={S}, DN={DN}")
    print(f"Tensor size u/V_post: {K*S*DN*2/1e6:.1f} MB per tensor\n")

    tests = [
        ("ONLY alloc V2 tensors",     only_alloc),
        ("ONLY fwd kernel (1 call)",  only_fwd_kernel),
        ("fwd+bwd kernel (1+1 call)", only_bwd_kernel),
        ("V1 full (fwd + bwd)",       v1_full),
        ("V2 full (fwd + bwd + re-fwd + bwd)", v2_full),
    ]

    for name, fn in tests:
        try:
            mean, std = time_it(fn, warmup=20, iters=50)
            print(f"  {name:<45s}  {mean:7.2f} ± {std:5.2f} ms")
        except Exception as e:
            print(f"  {name:<45s}  ERR: {e}")
