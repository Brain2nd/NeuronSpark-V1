"""NeuronLangevin 优化器 (Newton-Schulz 正交化动量 drift + Langevin 参数噪声 diffusion).

设计:
  drift: 对 2D 矩阵用 Newton-Schulz 正交化动量 (参考 Keller Jordan's Muon),
         对 1D/标量用 AdamW;
  diffusion: 可选的参数空间 Gaussian 噪声, 与 drift 正交叠加;
  合成更新:
      θ ← θ − η · drift_update(θ, ∇L) + √(2·η·T) · ξ,  ξ ~ N(0, I)

参考: https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def newton_schulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """5 阶 Newton-Schulz 迭代近似 G 的正交化: G ≈ UV^T (where G = UΣV^T).

    使用 Keller Jordan 的系数 (a, b, c) = (3.4445, -4.7750, 2.0315),
    使每步的多项式 p(σ) = a σ + b σ³ + c σ⁵ 把任意 σ ∈ [0, 1] 拉到 ≈1.
    要求输入已归一化到 spectral norm ≤ 1 (这里用 Frobenius norm 归一化, 更保守).
    """
    assert G.ndim >= 2, f"NS only for matrix params, got ndim={G.ndim}"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class NeuronLangevin(torch.optim.Optimizer):
    """Per-group dispatch 优化器: ortho-drift (Newton-Schulz) + AdamW-drift + Langevin diffusion.

    每个 param_group 必须声明 'optim_type' ∈ {'ortho', 'adamw'}.
    公共字段:
        lr:            基础学习率
        lr_mult:       学习率倍率 (默认 1.0)
        weight_decay:  decoupled weight decay (默认 0.0)
        temperature:   Langevin 温度 T (默认 0.0 即无噪声)
    ortho 专用:
        momentum:      动量系数 β (默认 0.95)
        ns_steps:      Newton-Schulz 迭代步数 (默认 5)
    AdamW 专用:
        betas:         (β1, β2) (默认 (0.9, 0.999))
        eps:           (默认 1e-8)
    """

    def __init__(
        self,
        params,
        lr: float = 2e-3,
        lr_mult: float = 1.0,
        weight_decay: float = 0.0,
        temperature: float = 0.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, lr_mult=lr_mult,
            weight_decay=weight_decay, temperature=temperature,
            momentum=momentum, ns_steps=ns_steps,
            betas=betas, eps=eps,
            optim_type='adamw',
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            opt = group.get('optim_type', 'adamw')
            if opt == 'ortho':
                self._step_ortho(group)
            elif opt == 'adamw':
                self._step_adamw(group)
            else:
                raise ValueError(f"unknown optim_type: {opt!r}")
        return loss

    # ============= ortho drift (Muon-style Newton-Schulz 正交化) =============
    def _step_ortho(self, group):
        lr = group['lr'] * group['lr_mult']
        beta = group['momentum']
        ns_steps = group['ns_steps']
        wd = group['weight_decay']
        T = group['temperature']

        for p in group['params']:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.float32)
            buf = state['momentum_buffer']
            buf.mul_(beta).add_(g.to(torch.float32))

            if p.ndim >= 2 and min(p.shape) >= 2:
                u = newton_schulz5(buf, steps=ns_steps)
                scale = math.sqrt(max(p.size(-2), p.size(-1)) / min(p.size(-2), p.size(-1)))
                u = u * scale
            elif p.ndim == 2 and min(p.shape) == 1:
                # D×1 或 1×D: 方向归一化 (Lion-like)
                flat = buf.reshape(-1)
                u = (flat / (flat.norm() + 1e-12)).reshape(buf.shape)
            else:
                # 1D 或标量: sign 更新
                u = buf.sign()

            if wd > 0:
                p.mul_(1.0 - lr * wd)
            p.add_(u.to(p.dtype), alpha=-lr)

            if T > 0:
                noise = torch.randn_like(p) * math.sqrt(2.0 * lr * T)
                p.add_(noise)

    # ================== AdamW drift ==================
    def _step_adamw(self, group):
        lr = group['lr'] * group['lr_mult']
        beta1, beta2 = group['betas']
        eps = group['eps']
        wd = group['weight_decay']
        T = group['temperature']

        for p in group['params']:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if 'step' not in state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
            state['step'] += 1
            t = state['step']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            g_fp32 = g.to(torch.float32)
            exp_avg.mul_(beta1).add_(g_fp32, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1 - beta2)

            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            denom = (exp_avg_sq / bc2).sqrt().add_(eps)
            update = (exp_avg / bc1) / denom

            if wd > 0:
                p.mul_(1.0 - lr * wd)
            p.add_(update.to(p.dtype), alpha=-lr)

            if T > 0:
                noise = torch.randn_like(p) * math.sqrt(2.0 * lr * T)
                p.add_(noise)


# ======================================================================
# Param group 构造器 (针对我们 v2.5 SNN 架构)
# ======================================================================

def _classify_param(name: str, p: torch.Tensor, ortho_min_dim: int) -> str:
    """返回 {'ortho_matrix', 'ortho_halt', 'adamw_plif', 'adamw_other'}."""
    # halt_proj: block_halt.weight / ffn_halt.weight, 形状 (1, D)
    if 'halt' in name:
        return 'ortho_halt'
    # PLIF / SNNBlock 1D 参数: .w / .v_th / .b_beta / .b_alpha / .b_th
    if name.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th')):
        return 'adamw_plif'
    # 2D 足够大 → ortho drift (Newton-Schulz)
    if p.ndim >= 2 and min(p.shape) >= ortho_min_dim:
        if 'embed_tokens' in name:  # token 稀疏梯度, NS 不适用
            return 'adamw_other'
        return 'ortho_matrix'
    # 其它 (RMSNorm / norm.gain / conv1d / 小矩阵 / bias) → AdamW
    return 'adamw_other'


def build_param_groups(
    model,
    *,
    ortho_lr: float = 2e-3,
    adamw_lr: float = 5e-5,
    plif_lr_mult: float = 10.0,
    weight_decay: float = 0.0,
    T_halt: float = 1e-3,
    T_ortho: float = 5e-5,
    T_plif: float = 1e-4,
    T_adamw_other: float = 0.0,
    momentum: float = 0.95,
    betas: tuple[float, float] = (0.9, 0.999),
    ortho_min_dim: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """把 SNNLanguageModel 的参数按功能/形状分成 4 组:

    ortho_matrix: 2D 矩阵 (所有 proj, 排除 embed_tokens) → Newton-Schulz drift + Langevin(T_ortho)
    ortho_halt:   block_halt / ffn_halt (1, D) → ortho drift (退化为方向归一化) + Langevin(T_halt)
    adamw_plif:   PLIF .w/.v_th + SNNBlock .b_beta/.b_alpha/.b_th → AdamW×plif_lr_mult + Langevin(T_plif)
    adamw_other:  embed_tokens + norm/RMSNorm + conv1d + 小矩阵 → AdamW + Langevin(T_adamw_other)

    直接扫 model.named_parameters(), 不依赖 model.get_param_groups() (后者漏 conv1d).
    """
    buckets = {
        'ortho_matrix': [],
        'ortho_halt': [],
        'adamw_plif': [],
        'adamw_other': [],
    }
    classification = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        tag = _classify_param(name, p, ortho_min_dim)
        buckets[tag].append(p)
        classification.append((name, tuple(p.shape), tag))

    if verbose:
        print("param classification:")
        for name, shape, tag in classification:
            print(f"  [{tag:15s}] {name}  shape={shape}")

    seen = set()
    for lst in buckets.values():
        for p in lst:
            pid = id(p)
            assert pid not in seen, "duplicate assignment"
            seen.add(pid)
    all_model = [p for p in model.parameters() if p.requires_grad]
    assert len(all_model) == len(seen), (
        f"param coverage mismatch: model has {len(all_model)} trainable params, "
        f"groups cover {len(seen)}"
    )

    return [
        dict(params=buckets['ortho_matrix'], optim_type='ortho', tag='ortho_matrix',
             lr=ortho_lr, lr_mult=1.0,
             weight_decay=weight_decay, temperature=T_ortho,
             momentum=momentum),
        dict(params=buckets['ortho_halt'], optim_type='ortho', tag='ortho_halt',
             lr=ortho_lr, lr_mult=1.0,
             weight_decay=0.0, temperature=T_halt,
             momentum=momentum),
        dict(params=buckets['adamw_plif'], optim_type='adamw', tag='adamw_plif',
             lr=adamw_lr, lr_mult=plif_lr_mult,
             weight_decay=0.0, temperature=T_plif,
             betas=betas),
        dict(params=buckets['adamw_other'], optim_type='adamw', tag='adamw_other',
             lr=adamw_lr, lr_mult=1.0,
             weight_decay=weight_decay, temperature=T_adamw_other,
             betas=betas),
    ]
