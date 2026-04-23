"""Moonshot-scaled Muon optimizer (vendored from KellerJordan/Muon with the
Moonshot-team LR scaling swap).

The ONLY difference from the upstream KellerJordan Muon is one line in the
Newton-Schulz post-processing:

    KellerJordan:  update *= max(1, m / n) ** 0.5       # rectangular compensation
    Moonshot    :  update *= 0.2 * max(m, n) ** 0.5     # absolute RMS ≈ AdamW

Moonshot's paper ("Muon is Scalable for LLM Training", 2025-02) argues that
choosing the scaling to match AdamW's per-element update RMS (≈0.2) lets you
re-use AdamW-tuned hyperparameters (LR, weight decay) when switching to Muon —
i.e. Muon becomes "drop-in scalable" without retuning.

Everything else — param-group structure, `use_muon` flag, distributed
round-robin + `all_gather`, momentum, weight decay — is identical to upstream.

Source (upstream): pip install git+https://github.com/KellerJordan/Muon
Moonshot reference: examples/toy_train.py in https://github.com/MoonshotAI/Moonlight
"""
from __future__ import annotations

import torch
import torch.distributed as dist


# ============================================================
# Newton-Schulz orthogonalization (unchanged from upstream)
# ============================================================

def zeropower_via_newtonschulz5(G, steps: int):
    """Quintic Newton-Schulz iteration to compute the zeroth-power (orthogonalization)."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ============================================================
# Muon update with Moonshot scaling
# ============================================================

def moonshot_muon_update(grad, momentum, beta: float = 0.95,
                         ns_steps: int = 5, nesterov: bool = True):
    """Muon update with Moonshot-team absolute RMS-match scaling.

    Replaces upstream's `update *= max(1, m/n)**0.5` with `update *= 0.2 * max(m, n)**0.5`.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # 4-D conv weights → view as 2-D
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # --- Moonshot scaling: absolute RMS ≈ 0.2 (matches AdamW per-element RMS) ---
    rows, cols = update.size(-2), update.size(-1)
    update *= 0.2 * (max(rows, cols) ** 0.5)
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard AdamW update (identical to upstream)."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# ============================================================
# Moonshot-scaled MuonWithAuxAdam (distributed)
# ============================================================

class MoonshotMuonWithAuxAdam(torch.optim.Optimizer):
    """Drop-in API-compatible replacement for upstream `MuonWithAuxAdam`
    with Moonshot's absolute LR scaling on Muon-flagged groups.

    Usage is IDENTICAL to upstream:
        groups = [
            {"params": matrix_params, "lr": ..., "momentum": 0.95,
             "weight_decay": 0, "use_muon": True},
            {"params": embed_params, "lr": ..., "betas": (0.9, 0.95),
             "eps": 1e-10, "weight_decay": 0, "use_muon": False},
            ...
        ]
        optimizer = MoonshotMuonWithAuxAdam(groups)

    Supports DeepSpeed ZeRO-0 round-robin + all_gather for distributed training
    (inherited pattern from upstream).
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                pad_n = (world_size - len(params) % world_size) % world_size
                params_pad = params + [torch.empty_like(params[-1])] * pad_n
                for base_i in range(0, len(params), world_size):
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = moonshot_muon_update(
                            p.grad, state["momentum_buffer"], beta=group["momentum"]
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    if world_size > 1:
                        dist.all_gather(
                            params_pad[base_i:base_i + world_size],
                            params_pad[base_i + rank],
                        )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMoonshotMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed variant of MoonshotMuonWithAuxAdam (for unit tests / single-GPU smoke)."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = moonshot_muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
