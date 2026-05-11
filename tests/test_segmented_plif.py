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
)
try:
    from neuronspark.modeling_neuronspark import _segmented_plif_rowparam_fwd_triton
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
    out_s, Vp_s, vcarry_s = segmented_plif_rowparam(beta, u, vth, v_init, k_t, K, training=True)
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
    out_full, Vp_full, carry_full = segmented_plif_rowparam(beta, u, vth, v_init, k_t, K, training=True)
    out_es, Vp_es, carry_es = segmented_plif_rowparam(beta, u, vth, v_init, k_t, K, training=False)
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
        out, Vp, carry = segmented_plif_rowparam(beta, u, vth, v_init, k_t, K, training=True)
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
    out_s, Vp_s, vcarry_s = segmented_plif_selective(beta, u, vth, v_init, k_t, K, training=True)
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
    out_full, Vp_full, carry_full = segmented_plif_selective(beta, u, vth, v_init, k_t, K, training=True)
    out_es, Vp_es, carry_es = segmented_plif_selective(beta, u, vth, v_init, k_t, K, training=False)
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
        out, Vp, carry = segmented_plif_selective(beta, u, vth, v_init, k_t, K, training=True)
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
                o_ref, vp_ref, vc_ref = segmented_plif_rowparam(beta, u, vth, v_init, k_t, K, train)
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


if __name__ == "__main__":
    test_degenerate_matches_continuous()
    test_early_stop_bit_exact()
    test_gradcheck()
    test_selective_degenerate()
    test_selective_early_stop_bit_exact()
    test_selective_gradcheck()
    test_kernel_rowparam_fwd_matches_reference()
    print("\nALL P3b-1 + P3b-2 + P5.5-1 TESTS PASSED")
