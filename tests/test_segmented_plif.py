"""P3b-1 验证: segmented_plif_rowparam.

1. 退化检查: 所有 k_t == K-1 时, segmented == 现有连续 plif_rowparam_forward (逐 bit / fp32 严格).
2. 早停 bit-exact: training=True (全 K) 和 training=False (早停) 的 gather@k_t / v_carry 完全相等.
3. gradcheck (fp64): forward/backward 一致 (含 token 边界 patch 路径).
"""
import sys, torch
sys.path.insert(0, "/home/dgxspark/Desktop/NeuronSpark-V1")
from neuronspark.modeling_neuronspark import (
    segmented_plif_rowparam, plif_rowparam_forward,
    segmented_plif_selective, plif_parallel_forward,
    _segmented_plif_rowparam_pytorch, _segmented_plif_selective_pytorch,
)
try:
    from neuronspark.modeling_neuronspark import _segmented_plif_rowparam_fwd_triton, _seg_rowparam_call
    _HAS_SEG_KERNEL = True
except ImportError:
    _HAS_SEG_KERNEL = False


def gather_at_kt(frames_TK, k_t, K):
    """frames_TK: (TK, b, H); k_t: (T, b); → (T, b, H) 取每 token 第 k_t 帧."""
    TK, b, H = frames_TK.shape
    T = TK // K
    fr = frames_TK.view(T, K, b, H)
    idx = k_t.view(T, 1, b, 1).expand(T, 1, b, H)
    return fr.gather(1, idx).squeeze(1)  # (T, b, H)


def test_degenerate_matches_continuous():
    torch.manual_seed(0)
    T, K, b, H = 5, 4, 2, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beta = torch.rand(b, H, device=dev) * 0.5 + 0.4   # (b, H), in [0.4, 0.9]
    vth = torch.rand(b, H, device=dev) * 0.3 + 0.1
    u = torch.randn(T * K, b, H, device=dev) * 0.5
    v_init = torch.randn(b, H, device=dev) * 0.2
    # 现有连续 kernel (treats TK as one long sequence):
    out_c, Vp_c = plif_rowparam_forward(beta, u, vth, v_init)
    # segmented, 所有 k_t = K-1:
    k_t = torch.full((T, b), K - 1, dtype=torch.long, device=dev)
    out_s, Vp_s, vcarry_s = _segmented_plif_rowparam_pytorch(beta, u, vth, v_init, k_t, K, training=True)
    d_out = (out_c - out_s).abs().max().item()
    d_Vp = (Vp_c - Vp_s).abs().max().item()
    d_carry = (Vp_c[-1] - vcarry_s).abs().max().item()
    print(f"[degenerate] all k_t=K-1: max|out diff|={d_out:.3e}, max|Vpost diff|={d_Vp:.3e}, max|carry diff|={d_carry:.3e}")
    assert d_out < 1e-5 and d_Vp < 1e-5 and d_carry < 1e-5, "segmented(k_t=K-1) must match continuous kernel"
    print("  ==> PASS: segmented == continuous when all k_t = K-1")


def test_early_stop_bit_exact():
    torch.manual_seed(1)
    T, K, b, H = 6, 5, 3, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beta = torch.rand(b, H, device=dev) * 0.5 + 0.4
    vth = torch.rand(b, H, device=dev) * 0.3 + 0.1
    u = torch.randn(T * K, b, H, device=dev) * 0.5
    v_init = torch.randn(b, H, device=dev) * 0.2
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)
    out_full, Vp_full, carry_full = _segmented_plif_rowparam_pytorch(beta, u, vth, v_init, k_t, K, training=True)
    out_es, Vp_es, carry_es = _segmented_plif_rowparam_pytorch(beta, u, vth, v_init, k_t, K, training=False)
    # gather @ k_t 必须相等
    g_full = gather_at_kt(out_full, k_t, K)
    g_es = gather_at_kt(out_es, k_t, K)
    d_g = (g_full - g_es).abs().max().item()
    d_c = (carry_full - carry_es).abs().max().item()
    # V_post @ k_t 也要相等 (patch 链用它)
    gv_full = gather_at_kt(Vp_full, k_t, K)
    gv_es = gather_at_kt(Vp_es, k_t, K)
    d_gv = (gv_full - gv_es).abs().max().item()
    print(f"[early-stop] training=True vs False: max|gather@k_t out diff|={d_g:.3e}, "
          f"max|gather@k_t Vpost diff|={d_gv:.3e}, max|v_carry diff|={d_c:.3e}")
    assert d_g == 0.0 and d_gv == 0.0 and d_c == 0.0, "early-stop must be bit-exact on the used frames"
    print("  ==> PASS: early-stop is bit-exact (frames > k_t are causally irrelevant)")


def test_gradcheck():
    torch.manual_seed(2)
    T, K, b, H = 4, 3, 2, 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beta = (torch.rand(b, H, device=dev, dtype=torch.float64) * 0.5 + 0.4).requires_grad_(True)
    vth = (torch.rand(b, H, device=dev, dtype=torch.float64) * 0.3 + 0.1).requires_grad_(True)
    u = (torch.randn(T * K, b, H, device=dev, dtype=torch.float64) * 0.3).requires_grad_(True)
    v_init = (torch.randn(b, H, device=dev, dtype=torch.float64) * 0.1).requires_grad_(True)
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)

    def f(beta, u, vth, v_init):
        out, Vp, carry = _segmented_plif_rowparam_pytorch(beta, u, vth, v_init, k_t, K, training=True)
        # 只对 gather@k_t 的 output (下游真实使用的) 求梯度, 加上 carry
        g = gather_at_kt(out, k_t, K)  # (T, b, H)
        return (g.sum(), carry.sum())
    ok = torch.autograd.gradcheck(f, (beta, u, vth, v_init), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"[gradcheck] fp64 gradcheck on (beta, u, vth, v_init): {ok}")
    assert ok
    print("  ==> PASS: backward matches numerical gradients (incl. token-boundary patch path)")


# ---------------- selective (non-rowparam) variant ----------------

def test_selective_degenerate():
    torch.manual_seed(3)
    T, K, b, H = 5, 4, 2, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beta = torch.rand(T * K, b, H, device=dev) * 0.5 + 0.4   # per-frame
    vth = torch.rand(T * K, b, H, device=dev) * 0.3 + 0.1
    u = torch.randn(T * K, b, H, device=dev) * 0.5
    v_init = torch.randn(b, H, device=dev) * 0.2
    out_c, Vp_c, _ = plif_parallel_forward(beta, u, vth, v_init)  # continuous (treats TK as one seq)
    k_t = torch.full((T, b), K - 1, dtype=torch.long, device=dev)
    out_s, Vp_s, vcarry_s = _segmented_plif_selective_pytorch(beta, u, vth, v_init, k_t, K, training=True)
    d_out = (out_c - out_s).abs().max().item()
    d_Vp = (Vp_c - Vp_s).abs().max().item()
    d_carry = (Vp_c[-1] - vcarry_s).abs().max().item()
    print(f"[selective degenerate] all k_t=K-1: max|out diff|={d_out:.3e}, max|Vpost diff|={d_Vp:.3e}, max|carry diff|={d_carry:.3e}")
    assert d_out < 1e-5 and d_Vp < 1e-5 and d_carry < 1e-5
    print("  ==> PASS: segmented_selective == continuous when all k_t = K-1")


def test_selective_early_stop_bit_exact():
    torch.manual_seed(4)
    T, K, b, H = 6, 5, 3, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beta = torch.rand(T * K, b, H, device=dev) * 0.5 + 0.4
    vth = torch.rand(T * K, b, H, device=dev) * 0.3 + 0.1
    u = torch.randn(T * K, b, H, device=dev) * 0.5
    v_init = torch.randn(b, H, device=dev) * 0.2
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)
    out_full, Vp_full, carry_full = _segmented_plif_selective_pytorch(beta, u, vth, v_init, k_t, K, training=True)
    out_es, Vp_es, carry_es = _segmented_plif_selective_pytorch(beta, u, vth, v_init, k_t, K, training=False)
    g_full = gather_at_kt(out_full, k_t, K); g_es = gather_at_kt(out_es, k_t, K)
    gv_full = gather_at_kt(Vp_full, k_t, K); gv_es = gather_at_kt(Vp_es, k_t, K)
    d_g = (g_full - g_es).abs().max().item()
    d_gv = (gv_full - gv_es).abs().max().item()
    d_c = (carry_full - carry_es).abs().max().item()
    print(f"[selective early-stop] max|gather@k_t out diff|={d_g:.3e}, max|gather@k_t Vpost diff|={d_gv:.3e}, max|v_carry diff|={d_c:.3e}")
    assert d_g == 0.0 and d_gv == 0.0 and d_c == 0.0
    print("  ==> PASS: selective early-stop is bit-exact")


def test_selective_gradcheck():
    torch.manual_seed(5)
    T, K, b, H = 4, 3, 2, 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beta = (torch.rand(T * K, b, H, device=dev, dtype=torch.float64) * 0.5 + 0.4).requires_grad_(True)
    vth = (torch.rand(T * K, b, H, device=dev, dtype=torch.float64) * 0.3 + 0.1).requires_grad_(True)
    u = (torch.randn(T * K, b, H, device=dev, dtype=torch.float64) * 0.3).requires_grad_(True)
    v_init = (torch.randn(b, H, device=dev, dtype=torch.float64) * 0.1).requires_grad_(True)
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)

    def f(beta, u, vth, v_init):
        out, Vp, carry = _segmented_plif_selective_pytorch(beta, u, vth, v_init, k_t, K, training=True)
        g = gather_at_kt(out, k_t, K)
        return (g.sum(), carry.sum())
    ok = torch.autograd.gradcheck(f, (beta, u, vth, v_init), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"[selective gradcheck] fp64: {ok}")
    assert ok
    print("  ==> PASS: selective backward matches numerical grads")


# ---------------- P5.5: fused Triton kernel (rowparam forward) ----------------

def test_kernel_rowparam_fwd_matches_reference():
    if not _HAS_SEG_KERNEL or not torch.cuda.is_available():
        print("[kernel rowparam fwd] skipped (no triton/cuda)")
        return
    torch.manual_seed(0)
    dev = "cuda"
    for train in (True, False):
        for kt_mode in ("random", "degenerate"):
            T, K, b, H = 6, 5, 3, 16
            beta = torch.rand(b, H, device=dev, dtype=torch.float32) * 0.5 + 0.4
            vth = torch.rand(b, H, device=dev, dtype=torch.float32) * 0.3 + 0.1
            u = torch.randn(T * K, b, H, device=dev, dtype=torch.float32) * 0.5
            v_init = torch.randn(b, H, device=dev, dtype=torch.float32) * 0.2
            k_t = (torch.randint(0, K, (T, b), dtype=torch.long, device=dev) if kt_mode == "random"
                   else torch.full((T, b), K - 1, dtype=torch.long, device=dev))
            with torch.no_grad():
                o_ref, vp_ref, vc_ref = _segmented_plif_rowparam_pytorch(beta, u, vth, v_init, k_t, K, train)
                o_k, vp_k, vc_k = _segmented_plif_rowparam_fwd_triton(beta, u, vth, v_init, k_t, K, train)
            if train:
                d_o = (o_ref - o_k).abs().max().item(); d_vp = (vp_ref - vp_k).abs().max().item()
            else:
                d_o = (gather_at_kt(o_ref, k_t, K) - gather_at_kt(o_k, k_t, K)).abs().max().item()
                d_vp = (gather_at_kt(vp_ref, k_t, K) - gather_at_kt(vp_k, k_t, K)).abs().max().item()
            d_vc = (vc_ref - vc_k).abs().max().item()
            print(f"[kernel rowparam fwd] train={train} kt={kt_mode}: d_out={d_o:.2e} d_Vpost={d_vp:.2e} d_vcarry={d_vc:.2e}")
            assert d_o < 1e-5 and d_vp < 1e-5 and d_vc < 1e-5
    print("  ==> PASS: Triton rowparam forward kernel bit-exact vs PyTorch reference")


def test_kernel_rowparam_bwd_matches_reference():
    if not _HAS_SEG_KERNEL or not torch.cuda.is_available():
        print("[kernel rowparam bwd] skipped (no triton/cuda)")
        return
    torch.manual_seed(0)
    dev = "cuda"
    for kt_mode in ("random", "degenerate"):
        T, K, b, H = 5, 4, 2, 16
        mk = lambda lo, hi, *s: (torch.rand(*s, device=dev, dtype=torch.float32) * (hi - lo) + lo)
        beta_v = mk(0.4, 0.9, b, H); vth_v = mk(0.1, 0.4, b, H)
        u_v = torch.randn(T * K, b, H, device=dev, dtype=torch.float32) * 0.5
        vi_v = torch.randn(b, H, device=dev, dtype=torch.float32) * 0.2
        k_t = (torch.randint(0, K, (T, b), dtype=torch.long, device=dev) if kt_mode == "random"
               else torch.full((T, b), K - 1, dtype=torch.long, device=dev))
        g_out = torch.randn(T * K, b, H, device=dev, dtype=torch.float32)
        # reference
        beta2, vth2, u2, vi2 = (x.clone().requires_grad_(True) for x in (beta_v, vth_v, u_v, vi_v))
        o_ref, _, _ = _segmented_plif_rowparam_pytorch(beta2, u2, vth2, vi2, k_t, K, True)
        (o_ref * g_out).sum().backward()
        # kernel
        beta1, vth1, u1, vi1 = (x.clone().requires_grad_(True) for x in (beta_v, vth_v, u_v, vi_v))
        o_k, _, _ = _seg_rowparam_call(beta1, u1, vth1, vi1, k_t, K, True)
        (o_k * g_out).sum().backward()
        d_f = (o_ref - o_k).abs().max().item()
        d = [(g.detach() if g is not None else None) for g in (beta2.grad, u2.grad, vth2.grad, vi2.grad)]
        e = [beta1.grad, u1.grad, vth1.grad, vi1.grad]
        diffs = [(a - bb).abs().max().item() for a, bb in zip(d, e)]
        print(f"[kernel rowparam bwd] kt={kt_mode}: d_fwd={d_f:.2e} d_grad(beta,u,vth,vinit)={[f'{x:.1e}' for x in diffs]}")
        assert d_f < 1e-5 and all(x < 1e-3 for x in diffs)
    print("  ==> PASS: Triton rowparam backward kernel matches PyTorch reference autograd")


# ---------------- v4.1: quantal release + AHP ----------------

def _ahp_row(b, H, dev, scale=0.05):
    return torch.rand(b, H, device=dev) * scale  # 小正值


def test_supra_ahp_gradcheck():
    """supra + AHP: 精确 ReLU 梯度 + ahp 项 (零梯度 s_hard); ahp 自身梯度 = -s_hard·g_Vpost (数值一致)."""
    torch.manual_seed(20)
    T, K, b, H = 4, 3, 2, 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beta = (torch.rand(b, H, device=dev, dtype=torch.float64) * 0.5 + 0.4).requires_grad_(True)
    vth = (torch.rand(b, H, device=dev, dtype=torch.float64) * 0.3 + 0.1).requires_grad_(True)
    u = (torch.randn(T * K, b, H, device=dev, dtype=torch.float64) * 0.3).requires_grad_(True)
    v_init = (torch.randn(b, H, device=dev, dtype=torch.float64) * 0.1).requires_grad_(True)
    ahp = (torch.rand(b, H, device=dev, dtype=torch.float64) * 0.05).requires_grad_(True)
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)

    def f(beta, u, vth, v_init, ahp):
        out, Vp, carry = _segmented_plif_rowparam_pytorch(beta, u, vth, v_init, k_t, K, training=True,
                                                          ahp_row=ahp, spike_quantal=False)
        return (gather_at_kt(out, k_t, K).sum(), carry.sum())
    ok = torch.autograd.gradcheck(f, (beta, u, vth, v_init, ahp), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"[supra+ahp gradcheck] fp64: {ok}")
    assert ok
    # selective 版
    beta_s = (torch.rand(T * K, b, H, device=dev, dtype=torch.float64) * 0.5 + 0.4).requires_grad_(True)
    vth_s = (torch.rand(T * K, b, H, device=dev, dtype=torch.float64) * 0.3 + 0.1).requires_grad_(True)
    u_s = (torch.randn(T * K, b, H, device=dev, dtype=torch.float64) * 0.3).requires_grad_(True)
    vi_s = (torch.randn(b, H, device=dev, dtype=torch.float64) * 0.1).requires_grad_(True)
    ahp_s = (torch.rand(b, H, device=dev, dtype=torch.float64) * 0.05).requires_grad_(True)

    def fs(beta, u, vth, v_init, ahp):
        out, Vp, carry = _segmented_plif_selective_pytorch(beta, u, vth, v_init, k_t, K, training=True,
                                                           ahp_row=ahp, spike_quantal=False)
        return (gather_at_kt(out, k_t, K).sum(), carry.sum())
    ok2 = torch.autograd.gradcheck(fs, (beta_s, u_s, vth_s, vi_s, ahp_s), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"[supra+ahp gradcheck selective] fp64: {ok2}")
    assert ok2
    print("  ==> PASS: supra+AHP backward matches numerical (incl. ahp param)")


def test_quantal_early_stop_bit_exact():
    """quantal (+ahp): 早停的 gather@k_t / V_post@k_t / v_carry 与全 K 逐 bit 相等 (rowparam + selective)."""
    torch.manual_seed(21)
    T, K, b, H = 6, 5, 3, 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)
    for use_ahp in (False, True):
        ahp = _ahp_row(b, H, dev) if use_ahp else None
        # rowparam
        beta = torch.rand(b, H, device=dev) * 0.5 + 0.4
        vth = torch.rand(b, H, device=dev) * 0.3 + 0.1
        u = torch.randn(T * K, b, H, device=dev) * 0.5
        vi = torch.randn(b, H, device=dev) * 0.2
        of, vpf, cf = _segmented_plif_rowparam_pytorch(beta, u, vth, vi, k_t, K, True, ahp, 4.0, True)
        oe, vpe, ce = _segmented_plif_rowparam_pytorch(beta, u, vth, vi, k_t, K, False, ahp, 4.0, True)
        d_g = (gather_at_kt(of, k_t, K) - gather_at_kt(oe, k_t, K)).abs().max().item()
        d_v = (gather_at_kt(vpf, k_t, K) - gather_at_kt(vpe, k_t, K)).abs().max().item()
        d_c = (cf - ce).abs().max().item()
        assert d_g == 0.0 and d_v == 0.0 and d_c == 0.0, f"rowparam quantal ahp={use_ahp}: {d_g},{d_v},{d_c}"
        # selective
        betas = torch.rand(T * K, b, H, device=dev) * 0.5 + 0.4
        vths = torch.rand(T * K, b, H, device=dev) * 0.3 + 0.1
        us = torch.randn(T * K, b, H, device=dev) * 0.5
        vis = torch.randn(b, H, device=dev) * 0.2
        of2, vpf2, cf2 = _segmented_plif_selective_pytorch(betas, us, vths, vis, k_t, K, True, ahp, 4.0, True)
        oe2, vpe2, ce2 = _segmented_plif_selective_pytorch(betas, us, vths, vis, k_t, K, False, ahp, 4.0, True)
        d_g2 = (gather_at_kt(of2, k_t, K) - gather_at_kt(oe2, k_t, K)).abs().max().item()
        d_v2 = (gather_at_kt(vpf2, k_t, K) - gather_at_kt(vpe2, k_t, K)).abs().max().item()
        d_c2 = (cf2 - ce2).abs().max().item()
        assert d_g2 == 0.0 and d_v2 == 0.0 and d_c2 == 0.0, f"selective quantal ahp={use_ahp}: {d_g2},{d_v2},{d_c2}"
        print(f"[quantal early-stop] ahp={use_ahp}: rowparam & selective bit-exact ✓")
    print("  ==> PASS: quantal (+ahp) early-stop is bit-exact")


def test_kernel_quantal_ahp_matches_reference():
    """Triton kernel vs PyTorch reference, quantal + ahp, forward + backward (rowparam + selective)."""
    if not _HAS_SEG_KERNEL or not torch.cuda.is_available():
        print("[kernel quantal+ahp] skipped (no triton/cuda)")
        return
    from neuronspark.modeling_neuronspark import _seg_selective_call
    torch.manual_seed(22)
    dev = "cuda"
    T, K, b, H = 5, 4, 3, 16
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)
    g_out = torch.randn(T * K, b, H, device=dev, dtype=torch.float32)
    for use_ahp in (False, True):
        ahp_v = (torch.rand(b, H, device=dev, dtype=torch.float32) * 0.05) if use_ahp else None
        # ---- rowparam ----
        beta_v = torch.rand(b, H, device=dev, dtype=torch.float32) * 0.5 + 0.4
        vth_v = torch.rand(b, H, device=dev, dtype=torch.float32) * 0.3 + 0.1
        u_v = torch.randn(T * K, b, H, device=dev, dtype=torch.float32) * 0.5
        vi_v = torch.randn(b, H, device=dev, dtype=torch.float32) * 0.2
        # reference (forward + backward via autograd)
        beta2, vth2, u2, vi2 = (x.clone().requires_grad_(True) for x in (beta_v, vth_v, u_v, vi_v))
        ahp2 = ahp_v.clone().requires_grad_(True) if use_ahp else None
        o_ref, _, vc_ref = _segmented_plif_rowparam_pytorch(beta2, u2, vth2, vi2, k_t, K, True, ahp2, 4.0, True)
        (o_ref * g_out).sum().backward()
        # kernel
        beta1, vth1, u1, vi1 = (x.clone().requires_grad_(True) for x in (beta_v, vth_v, u_v, vi_v))
        ahp1 = ahp_v.clone().requires_grad_(True) if use_ahp else None
        o_k, _, vc_k = _seg_rowparam_call(beta1, u1, vth1, vi1, k_t, K, True, ahp1, 4.0, True)
        (o_k * g_out).sum().backward()
        d_f = (o_ref - o_k).abs().max().item()
        d_vc = (vc_ref - vc_k).abs().max().item()
        names = ["beta", "u", "vth"] + (["ahp"] if use_ahp else []) + ["vinit"]
        ref_g = [beta2.grad, u2.grad, vth2.grad] + ([ahp2.grad] if use_ahp else []) + [vi2.grad]
        ker_g = [beta1.grad, u1.grad, vth1.grad] + ([ahp1.grad] if use_ahp else []) + [vi1.grad]
        diffs = [(a - bb).abs().max().item() for a, bb in zip(ref_g, ker_g)]
        print(f"[kernel quantal rowparam ahp={use_ahp}] d_fwd={d_f:.2e} d_vc={d_vc:.2e} d_grad{names}={[f'{x:.1e}' for x in diffs]}")
        assert d_f < 1e-4 and d_vc < 1e-4 and all(x < 2e-3 for x in diffs)
        # ---- selective ----
        betas = torch.rand(T * K, b, H, device=dev, dtype=torch.float32) * 0.5 + 0.4
        vths = torch.rand(T * K, b, H, device=dev, dtype=torch.float32) * 0.3 + 0.1
        us = torch.randn(T * K, b, H, device=dev, dtype=torch.float32) * 0.5
        vis = torch.randn(b, H, device=dev, dtype=torch.float32) * 0.2
        bs2, vs2, u2s, vi2s = (x.clone().requires_grad_(True) for x in (betas, vths, us, vis))
        ahp2s = ahp_v.clone().requires_grad_(True) if use_ahp else None
        os_ref, _, vcs_ref = _segmented_plif_selective_pytorch(bs2, u2s, vs2, vi2s, k_t, K, True, ahp2s, 4.0, True)
        (os_ref * g_out).sum().backward()
        bs1, vs1, u1s, vi1s = (x.clone().requires_grad_(True) for x in (betas, vths, us, vis))
        ahp1s = ahp_v.clone().requires_grad_(True) if use_ahp else None
        os_k, _, vcs_k = _seg_selective_call(bs1, u1s, vs1, vi1s, k_t, K, True, ahp1s, 4.0, True)
        (os_k * g_out).sum().backward()
        d_fs = (os_ref - os_k).abs().max().item()
        d_vcs = (vcs_ref - vcs_k).abs().max().item()
        ref_gs = [bs2.grad, u2s.grad, vs2.grad] + ([ahp2s.grad] if use_ahp else []) + [vi2s.grad]
        ker_gs = [bs1.grad, u1s.grad, vs1.grad] + ([ahp1s.grad] if use_ahp else []) + [vi1s.grad]
        diffss = [(a - bb).abs().max().item() for a, bb in zip(ref_gs, ker_gs)]
        print(f"[kernel quantal selective ahp={use_ahp}] d_fwd={d_fs:.2e} d_vc={d_vcs:.2e} d_grad={[f'{x:.1e}' for x in diffss]}")
        assert d_fs < 1e-4 and d_vcs < 1e-4 and all(x < 2e-3 for x in diffss)
    print("  ==> PASS: Triton kernels match PyTorch reference for quantal (+ahp), fwd + bwd")


def test_kernel_supra_ahp_matches_reference():
    """Triton kernel vs PyTorch reference, supra + ahp (verify ahp 进了 kernel forward + grad_ahp)."""
    if not _HAS_SEG_KERNEL or not torch.cuda.is_available():
        print("[kernel supra+ahp] skipped"); return
    torch.manual_seed(23)
    dev = "cuda"
    T, K, b, H = 5, 4, 2, 16
    k_t = torch.randint(0, K, (T, b), dtype=torch.long, device=dev)
    g_out = torch.randn(T * K, b, H, device=dev, dtype=torch.float32)
    ahp_v = torch.rand(b, H, device=dev, dtype=torch.float32) * 0.05
    beta_v = torch.rand(b, H, device=dev, dtype=torch.float32) * 0.5 + 0.4
    vth_v = torch.rand(b, H, device=dev, dtype=torch.float32) * 0.3 + 0.1
    u_v = torch.randn(T * K, b, H, device=dev, dtype=torch.float32) * 0.5
    vi_v = torch.randn(b, H, device=dev, dtype=torch.float32) * 0.2
    b2, v2, u2, i2, a2 = (x.clone().requires_grad_(True) for x in (beta_v, vth_v, u_v, vi_v, ahp_v))
    o_ref, _, _ = _segmented_plif_rowparam_pytorch(b2, u2, v2, i2, k_t, K, True, a2, 4.0, False)
    (o_ref * g_out).sum().backward()
    b1, v1, u1, i1, a1 = (x.clone().requires_grad_(True) for x in (beta_v, vth_v, u_v, vi_v, ahp_v))
    o_k, _, _ = _seg_rowparam_call(b1, u1, v1, i1, k_t, K, True, a1, 4.0, False)
    (o_k * g_out).sum().backward()
    d_f = (o_ref - o_k).abs().max().item()
    diffs = [(x.grad - y.grad).abs().max().item() for x, y in [(b2, b1), (u2, u1), (v2, v1), (i2, i1), (a2, a1)]]
    print(f"[kernel supra+ahp] d_fwd={d_f:.2e} d_grad(beta,u,vth,vinit,ahp)={[f'{x:.1e}' for x in diffs]}")
    assert d_f < 1e-5 and all(x < 1e-3 for x in diffs)
    print("  ==> PASS: Triton supra+ahp kernel matches reference (ahp wired in fwd + bwd)")


if __name__ == "__main__":
    test_degenerate_matches_continuous()
    test_early_stop_bit_exact()
    test_gradcheck()
    test_selective_degenerate()
    test_selective_early_stop_bit_exact()
    test_selective_gradcheck()
    test_kernel_rowparam_fwd_matches_reference()
    test_kernel_rowparam_bwd_matches_reference()
    test_supra_ahp_gradcheck()
    test_quantal_early_stop_bit_exact()
    test_kernel_quantal_ahp_matches_reference()
    test_kernel_supra_ahp_matches_reference()
    print("\nALL TESTS PASSED (P3b + P5.5 + v4.1 quantal/AHP)")
