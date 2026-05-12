"""Moonshot-Muon (matrices) + Adam (embed/norm) + Lion (逐通道神经元参数 .w/.v_th/.ahp, 1D tensor) hybrid optimizer.

扩展 `utils/muon_moonshot.py:MoonshotMuonWithAuxAdam`，增加 Lion (Chen et al. 2023, "Symbolic
Discovery of Optimization Algorithms") aux path 给「逐通道神经元参数」（per-channel β/v_th/ahp，1D tensor；b_beta/b_alpha/b_th 已在 bias 重构中删除）—— 这些参数在 V4.1 quantal 模式下梯度通过 `∂out/∂V_pre = v_th·surrogate'`
只有 ~0.02 量级（比 supra 的 ~1 弱 50x），又稀疏（发放率低→大多数 batch 无 spike→零梯度，rare
spike → 小但有信息的梯度），属于"稀疏+小幅+带噪"的最不利情景。

为什么 Lion 适合这个情景：
  - 更新 = `lr · sign(β1·momentum + (1-β1)·grad)`，**与梯度幅度无关**（Adam 在 v_t 衰减到 ~0 之后
    再来一个小幅梯度时 m/sqrt(v) 数值不稳；Lion 只看 sign）。
  - 双 β：β1 控制当步 sign 决策（默认 0.9），β2 控制 momentum 状态衰减（默认 0.99）—— momentum
    保留长程方向信息，对零梯度 batch 鲁棒；sign(momentum) 在 rare nonzero grad 触发时给一致方向。
  - 实践上 sparse-gradient 场景 Lion 普遍优于 Adam（Chen et al. Figure 9）。

矩阵权重（W_in/W_out/...）继续用 Muon (Moonshot scaling), 因为它们梯度幅度正常, Muon 已验证好用。
Embedding / RMSNorm 继续用 Adam (无需替换)。

DeepSpeed 兼容：跟 MoonshotMuonWithAuxAdam 同模式 —— `dist.all_gather` 走 Muon path（matrices
跨 rank 分担 Newton-Schulz 计算）；Lion / Adam path 是 per-param，每 rank 独立更新（ZeRO-0
replicates 参数; 任何 rank 上同一 param 的 grad 是同步聚合后的→ 各 rank 更新结果一致）。

Param group 字段：
  Muon group: {params, lr, momentum, weight_decay, use_muon=True}
  Aux  group: {params, lr, betas, eps, weight_decay, use_muon=False, aux_kind="adam"|"lion"}
  (省略 aux_kind 等同 "adam"，向后兼容旧 MoonshotMuonWithAuxAdam 调用)
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from .muon_moonshot import (
    zeropower_via_newtonschulz5,
    moonshot_muon_update,
    adam_update,
)


# ============================================================
# Stochastic rounding (用于 bf16 参数: round-to-nearest 会把 < bf16 精度的更新吃掉,
# stochastic rounding 按残差比例随机进位 → 期望无偏 → 小更新能正确累积. 这是 bf16/fp8 低精度训练标配.)
# ============================================================

_INT32_MASK_LOW16 = -65536  # = 0xFFFF0000 in int32 two's complement

def _stochastic_round_to_bf16(x_fp32: torch.Tensor) -> torch.Tensor:
    """fp32 → bf16, 随机舍入. bf16 = fp32 高 16 bit; 把低 16 bit 加一个 [0, 2^16) 的随机数再截断:
    → 残差 r 的尾数以概率 r/2^16 进位 (对正负数都是"以 r/2^16 概率舍离 0") → E[结果] = x (无偏)."""
    xi = x_fp32.contiguous().view(torch.int32)
    rnd = torch.randint(0, 1 << 16, x_fp32.shape, dtype=torch.int32, device=x_fp32.device)
    sr = (xi + rnd) & torch.tensor(_INT32_MASK_LOW16, dtype=torch.int32, device=x_fp32.device)
    return sr.view(torch.float32).to(torch.bfloat16)


def _apply_step(p, update, lr: float):
    """p -= lr·update; p 是 bf16 时在 fp32 里算更新再 stochastic-round 回 bf16
    (否则 < bf16 精度的更新被 round-to-nearest 吃掉, 见上). 后续的 p.clamp_(min=cmin) (Lion 防漂负) 由 caller 照常做."""
    if p.dtype == torch.bfloat16:
        p.data.copy_(_stochastic_round_to_bf16(p.data.float().sub_(update.float(), alpha=lr)))
    else:
        p.add_(update, alpha=-lr)


# ============================================================
# Lion update (Chen et al. 2023)
# ============================================================

def lion_update(grad, momentum, beta1: float = 0.9, beta2: float = 0.99):
    """Lion update — sign-based, scale-invariant.

    Update step:
        update = sign(β1 · momentum + (1 - β1) · grad)
        momentum ← β2 · momentum + (1 - β2) · grad

    返回 `update`（让 caller 应用 `p -= lr · update`，方便和 Muon/Adam 同接口）。
    `momentum` in-place 更新。
    """
    update = (momentum * beta1 + grad * (1 - beta1)).sign_()
    momentum.lerp_(grad.to(momentum.dtype), 1 - beta2)   # grad 可能 bf16, momentum fp32 → lerp_ 要求同 dtype
    return update


def _aux_kind(group) -> str:
    """Get aux optimizer kind for non-Muon groups (default adam, back-compat)."""
    return group.get("aux_kind", "adam")


def _validate_group(group):
    """Check param group has all required keys for its kind."""
    if group["use_muon"]:
        # Muon group: 跟 MoonshotMuonWithAuxAdam 一致
        group.setdefault("lr", 0.02)
        group.setdefault("momentum", 0.95)
        group.setdefault("weight_decay", 0)
        allowed = {"params", "lr", "momentum", "weight_decay", "use_muon"}
        assert set(group.keys()) <= allowed, f"Muon group has unexpected keys: {set(group.keys()) - allowed}"
        return
    kind = _aux_kind(group)
    if kind == "adam":
        group.setdefault("lr", 3e-4)
        group.setdefault("betas", (0.9, 0.95))
        group.setdefault("eps", 1e-10)
        group.setdefault("weight_decay", 0)
        allowed = {"params", "lr", "betas", "eps", "weight_decay", "use_muon", "aux_kind"}
        assert set(group.keys()) <= allowed
    elif kind == "lion":
        group.setdefault("lr", 1e-4)
        group.setdefault("betas", (0.9, 0.99))  # Lion 推荐 β1=0.9 β2=0.99
        group.setdefault("weight_decay", 0)
        # clamp_min: 可选下界，每步更新后 p.clamp_(min=cmin)。设 1e-4 防 v_th/ahp 漂负数 (quantal 翻转 NaN)
        allowed = {"params", "lr", "betas", "weight_decay", "use_muon", "aux_kind", "clamp_min"}
        assert set(group.keys()) <= allowed
    else:
        raise ValueError(f"unknown aux_kind: {kind!r}")


# ============================================================
# Distributed (ZeRO-0 compatible): MoonshotMuonAdamLion
# ============================================================

class MoonshotMuonAdamLion(torch.optim.Optimizer):
    """Muon (matrices) + Adam (embed/norm) + Lion (逐通道神经元参数, 1D tensor), distributed.

    用法：
        groups = [
            {"params": matrix_params, "lr": ..., "momentum": 0.95, "weight_decay": 0,
             "use_muon": True},
            {"params": embed_params, "lr": ..., "betas": (0.9, 0.95), "eps": 1e-10,
             "weight_decay": 0, "use_muon": False, "aux_kind": "adam"},
            {"params": neuron_params, "lr": ..., "betas": (0.9, 0.99),
             "weight_decay": 0, "use_muon": False, "aux_kind": "lion"},
        ]
        optimizer = MoonshotMuonAdamLion(groups)
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group, "each group must specify use_muon"
            _validate_group(group)
            if group["use_muon"]:
                # sort by size descending → larger matrices first (better load balancing)
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
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
                        _apply_step(p, update.reshape(p.shape), group["lr"])
                    if world_size > 1:
                        dist.all_gather(
                            params_pad[base_i:base_i + world_size],
                            params_pad[base_i + rank],
                        )
                continue
            # ---- aux path (per-param, replicated under ZeRO-0) ----
            kind = _aux_kind(group)
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if kind == "adam":
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
                    _apply_step(p, update, group["lr"])
                elif kind == "lion":
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)  # fp32: Lion momentum EMA 累积小梯度, bf16 会丢
                    b1, b2 = group["betas"]
                    update = lion_update(p.grad, state["momentum"], beta1=b1, beta2=b2)
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    _apply_step(p, update, group["lr"])
                    # 可选: 钳到 [clamp_min, ...) 防止 v_th/ahp 漂到负数引发动力学翻转 → NaN
                    cmin = group.get("clamp_min", None)
                    if cmin is not None:
                        p.clamp_(min=cmin)
        return loss


# ============================================================
# Single-device (no dist) — for ablation / unit tests
# ============================================================

class SingleDeviceMoonshotMuonAdamLion(torch.optim.Optimizer):
    """Non-distributed variant. Same API as MoonshotMuonAdamLion."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            _validate_group(group)
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
                    _apply_step(p, update.reshape(p.shape), group["lr"])
                continue
            kind = _aux_kind(group)
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if kind == "adam":
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
                    _apply_step(p, update, group["lr"])
                elif kind == "lion":
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)  # fp32: Lion momentum EMA 累积小梯度, bf16 会丢
                    b1, b2 = group["betas"]
                    update = lion_update(p.grad, state["momentum"], beta1=b1, beta2=b2)
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    _apply_step(p, update, group["lr"])
                    cmin = group.get("clamp_min", None)
                    if cmin is not None:
                        p.clamp_(min=cmin)
        return loss


# ============================================================
# Param group builder (Muon + Adam + Lion)
# ============================================================

def build_muon_adam_lion_param_groups(
    model,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    adam_base_lr: float = 2e-4,
    adam_embed_lr: float = 2e-4,
    lion_lr: float = 1e-4,
    lion_betas: tuple = (0.9, 0.99),
    neuron_lr_mult: float = 1.0,
    weight_decay_muon: float = 0.0,
    weight_decay_adam: float = 0.0,
    weight_decay_lion: float = 0.0,
    clamp_pos_lion: float = 1e-4,
) -> list[dict]:
    """Build 5-group param layout: Muon matrices / Adam embed / Adam norm / Lion neuron-pos / Lion neuron-free.

    Neuron 参数拆成两 Lion 子组:
      - neuron-pos: `.v_th`, `.ahp` —— **必须 ≥ 0** (负值会翻转 quantal 动力学引发 NaN), 每步更新后 clamp_(min=clamp_pos_lion)
      - neuron-free: `.w`, `.b_beta`, `.b_alpha`, `.b_th` —— 无约束 (β = sigmoid(w) 需要 w 自由取值;
        b_* 是 modulation 偏置, 加 |..| 后才进 v_th(t))
    其它 (Muon/Adam) 跟 build_muon_param_groups 一致。`neuron_lr_mult` 应用到 lion_lr。
    """
    from .param_groups import _partition_params_for_muon

    muon_params, adam_embed, adam_norm, adam_neuron = _partition_params_for_muon(model)

    # 按 name 拆 neuron_pos (v_th + ahp) / neuron_free (其它)
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    neuron_pos, neuron_free = [], []
    for p in adam_neuron:
        nm = name_by_id.get(id(p), "")
        if nm.endswith((".v_th", ".ahp")):
            neuron_pos.append(p)
        else:
            neuron_free.append(p)

    groups: list[dict] = []
    if muon_params:
        groups.append({
            "params": muon_params,
            "lr": muon_lr,
            "momentum": muon_momentum,
            "weight_decay": weight_decay_muon,
            "use_muon": True,
        })
    if adam_embed:
        groups.append({
            "params": adam_embed,
            "lr": adam_embed_lr,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": weight_decay_adam,
            "use_muon": False,
            "aux_kind": "adam",
        })
    if adam_norm:
        groups.append({
            "params": adam_norm,
            "lr": adam_base_lr,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": weight_decay_adam,
            "use_muon": False,
            "aux_kind": "adam",
        })
    if neuron_pos:
        groups.append({
            "params": neuron_pos,
            "lr": lion_lr * neuron_lr_mult,
            "betas": lion_betas,
            "weight_decay": weight_decay_lion,
            "use_muon": False,
            "aux_kind": "lion",
            "clamp_min": clamp_pos_lion,
        })
    if neuron_free:
        groups.append({
            "params": neuron_free,
            "lr": lion_lr * neuron_lr_mult,
            "betas": lion_betas,
            "weight_decay": weight_decay_lion,
            "use_muon": False,
            "aux_kind": "lion",
        })
    return groups
