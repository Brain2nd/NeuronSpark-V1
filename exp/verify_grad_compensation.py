"""
验证蒸馏优化修改的梯度正确性:

1. Natural Gradient 补偿: b_beta 饱和区补偿倍率 ≈ 1/σ'(b_beta)
2. halt_proj 独立 grad clip: 裁剪不影响其他参数
3. E[K] EMA smooth: 梯度回传到 halt_proj
4. Adaptive β: 纯 Python float 系数, 不影响计算图
5. Forward 输出形状一致性

用法:
    TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas python exp/verify_grad_compensation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from atomic_ops.bio_ssm_layer import BioSSMLayer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # 验证用 float32, 避免精度问题


def test_natural_gradient_compensation():
    """验证 1: Natural Gradient 补偿在 sigmoid 饱和区的放大倍率。"""
    print("=" * 60)
    print("验证 1: Natural Gradient 补偿")
    print("=" * 60)

    D, N, K = 128, 4, 8
    layer = BioSSMLayer(D=D, N=N, K=K, num_layers=10, layer_idx=0, ek_floor=2.0)
    layer = layer.to(device, dtype)

    # 设置 b_beta 到饱和区: σ(4.6) ≈ 0.99
    layer.snn_block.b_beta.data.fill_(4.6)

    x = torch.randn(32, 2, D, device=device, dtype=dtype)
    layer.input_neuron.v = 0.
    layer.snn_block.hidden_neuron.v = 0.
    h_out, pc, efc = layer(x)
    loss = h_out.sum() + 0.01 * pc
    loss.backward()

    b_beta = layer.snn_block.b_beta
    grad_before = b_beta.grad.abs().mean().item()

    # 模拟补偿 (与 distill_wrapper_v3.py 一致)
    with torch.no_grad():
        beta = torch.sigmoid(b_beta.data)
        sigmoid_deriv = (beta * (1.0 - beta)).clamp(min=0.01)
        b_beta.grad.div_(sigmoid_deriv)
    grad_after = b_beta.grad.abs().mean().item()

    ratio = grad_after / max(grad_before, 1e-12)
    sigma_prime = (torch.sigmoid(torch.tensor(4.6)) * (1 - torch.sigmoid(torch.tensor(4.6)))).item()
    expected_ratio = 1.0 / sigma_prime

    print(f"  b_beta 值:       4.6 (σ(4.6) = {torch.sigmoid(torch.tensor(4.6)).item():.4f})")
    print(f"  σ'(4.6):         {sigma_prime:.6f}")
    print(f"  补偿前 |grad|:   {grad_before:.6e}")
    print(f"  补偿后 |grad|:   {grad_after:.6e}")
    print(f"  实际放大倍率:     {ratio:.1f}x")
    print(f"  理论放大倍率:     {expected_ratio:.1f}x (1/σ')")

    # 允许 20% 误差 (clamp 和数值精度)
    ok = abs(ratio - expected_ratio) / expected_ratio < 0.2
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'} (误差 {abs(ratio-expected_ratio)/expected_ratio*100:.1f}%)")

    # 同时验证 b_alpha 补偿
    layer.zero_grad()
    layer.input_neuron.v = 0.
    layer.snn_block.hidden_neuron.v = 0.
    h_out2, pc2, _ = layer(x)
    (h_out2.sum() + 0.01 * pc2).backward()

    b_alpha = layer.snn_block.b_alpha
    grad_alpha_before = b_alpha.grad.abs().mean().item()

    with torch.no_grad():
        softplus_deriv = torch.sigmoid(b_alpha.data).clamp(min=0.1)
        b_alpha.grad.div_(softplus_deriv)
    grad_alpha_after = b_alpha.grad.abs().mean().item()
    alpha_ratio = grad_alpha_after / max(grad_alpha_before, 1e-12)

    print(f"\n  b_alpha 补偿倍率: {alpha_ratio:.2f}x (softplus'=sigmoid, 通常 1~10x)")
    print(f"  b_alpha |grad| 补偿前/后: {grad_alpha_before:.6e} → {grad_alpha_after:.6e}")
    ok_alpha = alpha_ratio >= 1.0  # softplus 补偿至少 ≥ 1x
    print(f"  结果: {'✓ PASS' if ok_alpha else '✗ FAIL'}")
    return ok and ok_alpha


def test_halt_proj_clip():
    """验证 2: halt_proj 独立裁剪不影响其他参数。"""
    print("\n" + "=" * 60)
    print("验证 2: halt_proj 独立 grad clip")
    print("=" * 60)

    D, N, K = 128, 4, 8
    layer = BioSSMLayer(D=D, N=N, K=K, num_layers=10, layer_idx=0, ek_floor=2.0)
    layer = layer.to(device, dtype)

    x = torch.randn(32, 2, D, device=device, dtype=dtype)
    layer.input_neuron.v = 0.
    layer.snn_block.hidden_neuron.v = 0.
    h_out, pc, efc = layer(x)
    # 大 loss 制造大梯度
    loss = 1000.0 * h_out.sum() + 100.0 * pc
    loss.backward()

    # 记录裁剪前的梯度
    halt_grad_before = torch.cat([
        p.grad.flatten() for p in layer.halt_proj.parameters() if p.grad is not None
    ]).norm().item()
    out_proj_grad_before = layer.out_proj.weight.grad.norm().item()

    # 独立裁剪 halt_proj
    halt_params = [p for p in layer.halt_proj.parameters() if p.grad is not None]
    torch.nn.utils.clip_grad_norm_(halt_params, 0.5)

    halt_grad_after = torch.cat([
        p.grad.flatten() for p in layer.halt_proj.parameters() if p.grad is not None
    ]).norm().item()
    out_proj_grad_after = layer.out_proj.weight.grad.norm().item()

    print(f"  halt_proj grad norm: {halt_grad_before:.4f} → {halt_grad_after:.4f}")
    print(f"  out_proj grad norm:  {out_proj_grad_before:.4f} → {out_proj_grad_after:.4f} (不应变)")

    clipped = halt_grad_after <= 0.5 + 1e-4
    untouched = abs(out_proj_grad_before - out_proj_grad_after) < 1e-6
    print(f"  halt_proj 已裁剪:  {'✓ PASS' if clipped else '✗ FAIL'}")
    print(f"  out_proj 未受影响: {'✓ PASS' if untouched else '✗ FAIL'}")
    return clipped and untouched


def test_ek_smooth_gradient():
    """验证 3: ek_smooth_cost 梯度回传到 halt_proj。"""
    print("\n" + "=" * 60)
    print("验证 3: E[K] EMA smooth 梯度路径")
    print("=" * 60)

    D, N, K = 128, 4, 8
    layer = BioSSMLayer(D=D, N=N, K=K, num_layers=10, layer_idx=0, ek_floor=2.0)
    layer = layer.to(device, dtype)

    # 模拟 EMA = 5.0 (detached target)
    ema_target = 5.0

    x = torch.randn(32, 2, D, device=device, dtype=dtype)
    layer.input_neuron.v = 0.
    layer.snn_block.hidden_neuron.v = 0.
    h_out, ponder_cost, efc = layer(x)

    # ek_smooth = (ponder_cost - ema)^2
    ek_smooth = (ponder_cost - ema_target).pow(2)
    ek_smooth.backward()

    halt_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer.halt_proj.parameters()
    )
    # snn_block 参数不应有梯度 (只通过 ponder_cost)
    out_proj_grad = layer.out_proj.weight.grad
    out_proj_zero = out_proj_grad is None or out_proj_grad.abs().sum() < 1e-8

    print(f"  ponder_cost:       {ponder_cost.item():.4f}")
    print(f"  ek_smooth:         {ek_smooth.item():.4f}")
    print(f"  halt_proj 有梯度:  {'✓ PASS' if halt_has_grad else '✗ FAIL'}")
    print(f"  out_proj 无梯度:   {'✓ PASS' if out_proj_zero else '✗ FAIL'} (ek_smooth 只经过 halt)")
    return halt_has_grad and out_proj_zero


def test_adaptive_beta():
    """验证 4: adaptive β 不影响计算图 (纯 float 系数)。"""
    print("\n" + "=" * 60)
    print("验证 4: Adaptive β 计算图安全")
    print("=" * 60)

    from train_distill_v3 import get_adaptive_beta

    # cos < floor → boost
    beta1 = get_adaptive_beta(0.08, 2.0, 0.15)
    # cos >= floor → 不变
    beta2 = get_adaptive_beta(0.20, 2.0, 0.15)
    # cos = 0 → 最大 boost (3x)
    beta3 = get_adaptive_beta(0.0, 2.0, 0.15)

    print(f"  cos=0.08, β_base=2.0: → {beta1:.2f} (预期 ~2.0×{1+2*(1-0.08/0.15):.2f} = {2.0*(1+2*(1-0.08/0.15)):.2f})")
    print(f"  cos=0.20, β_base=2.0: → {beta2:.2f} (预期 2.00, 不放大)")
    print(f"  cos=0.00, β_base=2.0: → {beta3:.2f} (预期 6.00, 最大 3x)")

    ok1 = beta1 > 2.0  # 有放大
    ok2 = abs(beta2 - 2.0) < 1e-6  # 不放大
    ok3 = abs(beta3 - 6.0) < 1e-6  # 3x
    ok4 = isinstance(beta1, float)  # 纯 float

    print(f"  cos < floor 放大:  {'✓ PASS' if ok1 else '✗ FAIL'}")
    print(f"  cos >= floor 不变: {'✓ PASS' if ok2 else '✗ FAIL'}")
    print(f"  cos=0 最大 3x:     {'✓ PASS' if ok3 else '✗ FAIL'}")
    print(f"  返回 Python float: {'✓ PASS' if ok4 else '✗ FAIL'}")
    return ok1 and ok2 and ok3 and ok4


def test_forward_shape():
    """验证 5: Forward 输出形状和类型一致性。"""
    print("\n" + "=" * 60)
    print("验证 5: Forward 输出形状一致性")
    print("=" * 60)

    D, N, K = 128, 4, 8
    seq_len, batch = 32, 2
    layer = BioSSMLayer(D=D, N=N, K=K, num_layers=10, layer_idx=0, ek_floor=2.0)
    layer = layer.to(device, dtype)

    x = torch.randn(seq_len, batch, D, device=device, dtype=dtype)
    layer.input_neuron.v = 0.
    layer.snn_block.hidden_neuron.v = 0.
    h_out, pc, efc = layer(x)

    shape_ok = h_out.shape == (seq_len, batch, D)
    pc_scalar = pc.dim() == 0
    efc_scalar = efc.dim() == 0

    print(f"  输入:     ({seq_len}, {batch}, {D})")
    print(f"  h_out:    {tuple(h_out.shape)} {'✓' if shape_ok else '✗'}")
    print(f"  ponder:   scalar={pc_scalar}, val={pc.item():.4f} {'✓' if pc_scalar else '✗'}")
    print(f"  ek_floor: scalar={efc_scalar}, val={efc.item():.4f} {'✓' if efc_scalar else '✗'}")

    # 残差检查: h_out 不应 == x (有非零输出)
    diff = (h_out - x).abs().mean().item()
    has_output = diff > 1e-6
    print(f"  残差增量: {diff:.6f} {'✓ (非零)' if has_output else '✗ (全零，异常)'}")

    return shape_ok and pc_scalar and efc_scalar and has_output


def test_distill_schedule():
    """验证 6: 蒸馏调度 α_ce 修改生效。"""
    print("\n" + "=" * 60)
    print("验证 6: 蒸馏调度 α_ce Phase 1 = 0.7")
    print("=" * 60)

    from train_distill_v3 import get_distill_schedule

    a1, b1 = get_distill_schedule(0, 1000)        # Phase 1 起点
    a2, b2 = get_distill_schedule(299, 1000)       # Phase 1 末尾
    a3, b3 = get_distill_schedule(500, 1000)       # Phase 2 中间
    a4, b4 = get_distill_schedule(800, 1000)       # Phase 3

    print(f"  Phase 1 (step 0):   α={a1:.2f}, β={b1:.2f}")
    print(f"  Phase 1 (step 299): α={a2:.2f}, β={b2:.2f}")
    print(f"  Phase 2 (step 500): α={a3:.2f}, β={b3:.2f}")
    print(f"  Phase 3 (step 800): α={a4:.2f}, β={b4:.2f}")

    ok1 = abs(a1 - 0.7) < 1e-6  # Phase 1 α = 0.7
    ok2 = abs(a4 - 1.0) < 1e-6  # Phase 3 α = 1.0
    ok3 = abs(b1 - 2.0) < 1e-6  # Phase 1 β = 2.0
    ok4 = abs(b4 - 0.1) < 1e-6  # Phase 3 β = 0.1
    ok5 = 0.7 < a3 < 1.0        # Phase 2 过渡中

    print(f"  Phase 1 α=0.7: {'✓ PASS' if ok1 else '✗ FAIL'}")
    print(f"  Phase 3 α=1.0: {'✓ PASS' if ok2 else '✗ FAIL'}")
    print(f"  Phase 2 过渡:  {'✓ PASS' if ok5 else '✗ FAIL'} (α={a3:.2f})")
    return ok1 and ok2 and ok3 and ok4 and ok5


def test_curriculum_acceleration():
    """验证 7: 加速 curriculum 阈值。"""
    print("\n" + "=" * 60)
    print("验证 7: 加速 Curriculum 阈值")
    print("=" * 60)

    from train_distill_v3 import get_curriculum_seq_len

    total = 10000
    max_len = 512

    s1 = get_curriculum_seq_len(400, total, max_len)    # 4% → seq=64
    s2 = get_curriculum_seq_len(1000, total, max_len)   # 10% → seq=128
    s3 = get_curriculum_seq_len(2000, total, max_len)   # 20% → seq=256
    s4 = get_curriculum_seq_len(3000, total, max_len)   # 30% → seq=max

    print(f"  4% (step 400):   seq={s1} (预期 64)")
    print(f"  10% (step 1000): seq={s2} (预期 128)")
    print(f"  20% (step 2000): seq={s3} (预期 256)")
    print(f"  30% (step 3000): seq={s4} (预期 {max_len})")

    ok = (s1 == 64) and (s2 == 128) and (s3 == 256) and (s4 == max_len)
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


if __name__ == "__main__":
    torch.manual_seed(42)

    results = []
    results.append(("Natural Gradient 补偿", test_natural_gradient_compensation()))
    results.append(("halt_proj 独立 clip", test_halt_proj_clip()))
    results.append(("E[K] EMA smooth 梯度", test_ek_smooth_gradient()))
    results.append(("Adaptive β 安全", test_adaptive_beta()))
    results.append(("Forward 形状一致", test_forward_shape()))
    results.append(("蒸馏调度 α_ce", test_distill_schedule()))
    results.append(("Curriculum 加速", test_curriculum_acceleration()))

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print(f"\n{'全部通过!' if all_pass else '存在失败项!'}")
    sys.exit(0 if all_pass else 1)
