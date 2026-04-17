"""NeuronLangevin 单元测试 (本地 CPU/GPU 均可).

验证:
  1. param group 划分完整且无重复
  2. ortho drift 在 2D 矩阵上能推动参数
  3. halt_proj (1×D) 走 ortho 退化路径
  4. AdamW 分支在 1D 上正常
  5. Langevin 温度 T>0 时参数有额外噪声扰动
  6. 完整 SNNLanguageModel 上构造 param_groups 无遗漏
"""
from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from atomic_ops.neuron_langevin import (
    NeuronLangevin, newton_schulz5, build_param_groups,
)


def test_newton_schulz_shapes():
    for shape in [(64, 64), (128, 64), (64, 128), (1024, 1), (1, 1024)]:
        M = torch.randn(*shape)
        U = newton_schulz5(M, steps=5)
        assert U.shape == M.shape, f"shape mismatch {U.shape} vs {M.shape}"
        assert not torch.isnan(U).any(), f"NaN in NS output for shape {shape}"
    print("[PASS] newton_schulz_shapes")


def test_ortho_moves_matrix():
    """ortho drift 推动 2D 矩阵参数."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(128, 64))
    group = dict(
        params=[p], optim_type='ortho',
        lr=1e-2, lr_mult=1.0, momentum=0.95, ns_steps=5,
        weight_decay=0.0, temperature=0.0, betas=(0.9, 0.999), eps=1e-8,
    )
    opt = NeuronLangevin([group])
    p_init = p.detach().clone()

    for _ in range(5):
        opt.zero_grad()
        loss = (p ** 2).sum()
        loss.backward()
        opt.step()

    delta = (p - p_init).abs().mean().item()
    assert delta > 1e-3, f"ortho 未推动参数, Δ={delta}"
    print(f"[PASS] ortho_moves_matrix (Δ={delta:.4f})")


def test_ortho_halt_projection():
    """halt_proj (1, D) 走 ortho 的 2D 路径 (non-square)."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(1, 1024))  # nn.Linear(D, 1).weight shape
    group = dict(
        params=[p], optim_type='ortho',
        lr=5e-3, lr_mult=1.0, momentum=0.95, ns_steps=5,
        weight_decay=0.0, temperature=0.0, betas=(0.9, 0.999), eps=1e-8,
    )
    opt = NeuronLangevin([group])
    norm_init = p.norm().item()
    p_init = p.detach().clone()

    for _ in range(10):
        opt.zero_grad()
        loss = (p ** 2).sum()
        loss.backward()
        opt.step()

    delta = (p - p_init).abs().mean().item()
    norm_after = p.norm().item()
    assert delta > 1e-4, f"halt_proj 未动, Δ={delta}"
    print(f"[PASS] ortho_halt_projection (Δ={delta:.4e}, ‖p‖: {norm_init:.4f} → {norm_after:.4f})")


def test_adamw_moves_1d():
    """AdamW 分支更新 1D 参数."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(1024))
    group = dict(
        params=[p], optim_type='adamw',
        lr=1e-3, lr_mult=1.0, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, temperature=0.0, momentum=0.95, ns_steps=5,
    )
    opt = NeuronLangevin([group])
    p_init = p.detach().clone()

    for _ in range(5):
        opt.zero_grad()
        loss = (p ** 2).sum()
        loss.backward()
        opt.step()

    delta = (p - p_init).abs().mean().item()
    assert delta > 1e-4, f"AdamW 1D 未推动, Δ={delta}"
    print(f"[PASS] adamw_moves_1d (Δ={delta:.4e})")


def test_langevin_noise_kicks_frozen():
    """grad=0, drift=0, 但 T>0, 参数应仍被 Langevin 噪声推动."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.zeros(1024))
    group = dict(
        params=[p], optim_type='adamw',
        lr=1e-3, lr_mult=1.0, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, temperature=1.0,
        momentum=0.95, ns_steps=5,
    )
    opt = NeuronLangevin([group])
    p_init = p.detach().clone()

    for _ in range(20):
        opt.zero_grad()
        p.grad = torch.zeros_like(p)
        opt.step()

    delta = (p - p_init).std().item()
    expected = math.sqrt(2 * 1e-3 * 1.0 * 20)
    assert delta > 0.5 * expected, f"Langevin 噪声不足, 实测 std={delta:.4e}, 期望≈{expected:.4e}"
    print(f"[PASS] langevin_noise_kicks_frozen (std={delta:.4e}, 期望 ≈ {expected:.4e})")


def test_build_param_groups_on_real_model():
    """在 SNNLanguageModel 上构造 param_groups, 验证覆盖完整."""
    from model import SNNLanguageModel
    torch.manual_seed(0)
    model = SNNLanguageModel(
        vocab_size=128, D=64, N=2, K=3, num_layers=4, D_ff=128,
    )
    groups = build_param_groups(
        model,
        ortho_lr=2e-3, adamw_lr=5e-5, plif_lr_mult=10.0,
        weight_decay=0.0,
        T_halt=1e-3, T_ortho=5e-5, T_plif=1e-4, T_adamw_other=0.0,
    )

    total_params = 0
    total_tensors = 0
    print("\n  param groups:")
    for g in groups:
        n_params = sum(p.numel() for p in g['params'])
        n_tensors = len(g['params'])
        total_params += n_params
        total_tensors += n_tensors
        print(f"    [{g['tag']:15s}] tensors={n_tensors:4d}  params={n_params:>10,}  "
              f"optim={g['optim_type']}  lr={g['lr']:.0e}  T={g['temperature']:.0e}")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_tensors = len([p for p in model.parameters() if p.requires_grad])
    assert total_params == model_params, f"params: groups={total_params}, model={model_params}"
    assert total_tensors == model_tensors, f"tensors: groups={total_tensors}, model={model_tensors}"
    print(f"  合计: {total_tensors} 张量 / {total_params:,} 参数 (与 model 一致)")
    print("[PASS] build_param_groups_on_real_model")


def test_lr_mult_applied_once():
    """NeuronLangevin 内部应用 lr * lr_mult 一次; 外部 schedule 只动 lr, 不碰 lr_mult."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.ones(1024))  # 初始 ||p||^2 = 1024
    group = dict(
        params=[p], optim_type='adamw',
        lr=1e-4, lr_mult=10.0,  # effective = 1e-3
        betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, temperature=0.0, momentum=0.95, ns_steps=5,
    )
    opt = NeuronLangevin([group])

    # grad 全是 1 → 稳态下 update = lr * lr_mult * 1 ≈ 1e-3 每步
    for _ in range(3):
        opt.zero_grad()
        p.grad = torch.ones_like(p)
        opt.step()

    # p 从 1.0 → 大约 1.0 - 3 * 1e-3 = 0.997 (head-room: Adam bias correction 和 denom 收敛)
    mean_p = p.mean().item()
    # 允许 Adam warmup 带来的偏差, 但不应该跌到 0.97 级 (双 mult 就是 1e-2 每步)
    assert mean_p > 0.99, f"update too large, suspect double lr_mult. mean(p)={mean_p:.4f}"
    assert mean_p < 0.999, f"update too small. mean(p)={mean_p:.4f}"
    print(f"[PASS] lr_mult_applied_once (mean(p) after 3 steps = {mean_p:.4f}, 单次 mult 预期 ~0.997)")


def test_end_to_end_forward_backward():
    """完整 forward+backward+step 不报错."""
    from model import SNNLanguageModel
    torch.manual_seed(0)
    model = SNNLanguageModel(vocab_size=128, D=64, N=2, K=3, num_layers=4, D_ff=128)

    groups = build_param_groups(model,
        ortho_lr=2e-3, adamw_lr=5e-5, plif_lr_mult=10.0,
        T_halt=1e-3, T_ortho=5e-5, T_plif=1e-4, T_adamw_other=0.0,
    )
    opt = NeuronLangevin(groups)

    X = torch.randint(0, 128, (2, 16))
    Y = torch.randint(0, 128, (2, 16))
    out = model(X, Y)
    loss = out.last_loss.mean()
    loss.backward()
    opt.step()
    print(f"[PASS] end_to_end_forward_backward (loss={loss.item():.4f})")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    print("=" * 60)
    print("NeuronLangevin 单元测试")
    print("=" * 60)
    test_newton_schulz_shapes()
    test_ortho_moves_matrix()
    test_ortho_halt_projection()
    test_adamw_moves_1d()
    test_langevin_noise_kicks_frozen()
    test_lr_mult_applied_once()
    test_build_param_groups_on_real_model()
    test_end_to_end_forward_backward()
    print("\n" + "=" * 60)
    print("全部通过")
    print("=" * 60)
