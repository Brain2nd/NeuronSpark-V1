"""
NeuronSpark SNN Language Model — 单文件 HuggingFace 兼容实现。

本文件由 scripts/merge_modeling.py 自动生成，合并自:
  atomic_ops/snn_base.py, rms_norm.py, parallel_scan.py,
  plif_node.py, selective_plif.py, lateral_inhibition.py,
  snn_ffn.py, snn_block.py, snn_decoder_layer.py,
  snn_attention_decoder_layer.py, model.py + HF wrapper

不要手工编辑此文件。修改请在源文件中进行，然后重新运行合并脚本。
"""

import copy
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_neuronspark import NeuronSparkConfig

# Triton (可选 GPU 加速)
_SYSTEM_PTXAS = '/usr/local/cuda-13.0/bin/ptxas'
if os.path.exists(_SYSTEM_PTXAS) and 'TRITON_PTXAS_PATH' not in os.environ:
    os.environ['TRITON_PTXAS_PATH'] = _SYSTEM_PTXAS

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass



# ============================================================
# Section: snn_base
# ============================================================

# ============================================================
# MemoryModule
# ============================================================

class MemoryModule(nn.Module):
    """有状态模块基类：提供 register_memory / reset 状态管理。

    提供 register_memory / reset 状态管理:
    - _memories: {name: current_value}
    - _memories_rv: {name: reset_value (deepcopy)}
    - __getattr__ / __setattr__ / __delattr__ 代理 _memories
    - reset() 用 deepcopy(_memories_rv[name]) 还原
    """

    def __init__(self):
        super().__init__()
        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        """注册有状态变量。reset() 时会被还原为 value 的 deepcopy。"""
        self._memories[name] = value
        self._memories_rv[name] = copy.deepcopy(value)

    def reset(self):
        """重置所有有状态变量为注册时的值。"""
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value):
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            super().__delattr__(name)

# ============================================================
# BaseNode
# ============================================================

class BaseNode(MemoryModule):
    """脉冲神经元基类：膜电位状态 + v_float_to_tensor。

    保留当前仓库实际使用的全部构造参数和属性。
    """

    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function=None, detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq

    def v_float_to_tensor(self, x: torch.Tensor):
        """将标量膜电位扩展为与输入同形的张量。"""
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

# ============================================================
# Sigmoid
# ============================================================

class _SigmoidGrad(torch.autograd.Function):
    """Heaviside forward + sigmoid surrogate backward。"""

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x)  # 保留输入 dtype

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        sgax = (x * alpha).sigmoid_()
        return grad_output * (1. - sgax) * sgax * alpha, None

class Sigmoid(nn.Module):
    """Surrogate gradient: Heaviside forward, sigmoid backward.

    类名必须是 Sigmoid — parallel_scan.py 按 type().__name__ 判断 Triton 快路径。
    """

    def __init__(self, alpha=4.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return _SigmoidGrad.apply(x, self.alpha)
        else:
            return (x * self.alpha).sigmoid()

# ============================================================
# Linear
# ============================================================

class Linear(nn.Linear):
    """nn.Linear + step_mode 属性（保留接口契约）。"""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, step_mode='s'):
        super().__init__(in_features, out_features, bias)
        self.step_mode = step_mode

# ============================================================
# reset_net
# ============================================================

def reset_net(net: nn.Module):
    """重置网络中所有 MemoryModule 的状态。"""
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()

# ============================================================
# 命名空间导出 — 让下游代码零改动
# ============================================================

class surrogate:
    """命名空间：surrogate.Sigmoid(alpha=4.0) 保持原调用方式。"""
    Sigmoid = Sigmoid

class functional:
    """命名空间：functional.reset_net(net) 保持原调用方式。"""
    reset_net = staticmethod(reset_net)

class layer:
    """命名空间：layer.Linear(D, D_ff, bias=False, step_mode='s') 保持原调用方式。"""
    Linear = Linear


# ============================================================
# Section: rms_norm
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    x_norm = x / RMS(x) * weight
    RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: 归一化维度
        eps: 数值稳定性
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


# ============================================================
# Section: parallel_scan
# ============================================================

# ============================================================
# Triton fused recurrence kernels
# ============================================================

# DGX Spark (GB10, sm_121a): Triton 3.5.1 自带 ptxas 不支持 sm_121a，
# 需要使用系统 CUDA 13.0 的 ptxas
if _HAS_TRITON:

    @triton.jit
    def _fwd_recurrence_kernel(
        A_ptr, B_ptr, INIT_ptr, OUT_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Forward: V[k] = A[k]*V[k-1] + B[k], V[-1] = init.

        Grid: (ceil(num_cols / BLOCK),)
        Each program processes BLOCK columns across all K sequential steps.
        Accumulation in float32; storage in input dtype.
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            a = tl.load(A_ptr + off, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptr + off, mask=mask, other=0.0).to(tl.float32)
            v = a * v + b
            tl.store(OUT_ptr + off, v, mask=mask)

    @triton.jit
    def _bwd_recurrence_kernel(
        A_ptr, V_ptr, INIT_ptr, GRAD_OUT_ptr,
        GRAD_A_ptr, GRAD_B_ptr, GRAD_INIT_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Backward for V[k] = A[k]*V[k-1] + B[k].

        Reverse accumulation (k from K-1 down to 0):
          g = 0
          for k = K-1, ..., 0:
            g += grad_out[k]
            grad_B[k] = g
            grad_A[k] = g * V[k-1]   (V[-1] = init)
            g = g * A[k]
          grad_init = g
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        g = tl.zeros([BLOCK], dtype=tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            dV = tl.load(GRAD_OUT_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = g + dV

            tl.store(GRAD_B_ptr + off, g, mask=mask)

            if k > 0:
                v_prev = tl.load(
                    V_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
            else:
                v_prev = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(GRAD_A_ptr + off, g * v_prev, mask=mask)

            a = tl.load(A_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = g * a

        tl.store(GRAD_INIT_ptr + cols, g, mask=mask)

    class _TritonLinearRecurrence(torch.autograd.Function):
        """Fused Triton linear recurrence: V[k] = A[k]*V[k-1] + B[k]."""

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta, u, v_init):
            beta_c = beta.contiguous()
            u_c = u.contiguous()
            v_init_c = v_init.contiguous()

            K = beta_c.shape[0]
            num_cols = beta_c[0].numel()
            V = torch.empty_like(u_c)

            BLOCK = _TritonLinearRecurrence._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fwd_recurrence_kernel[grid](
                beta_c, u_c, v_init_c, V,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                ctx.save_for_backward(beta_c, V, v_init_c)
            ctx.K = K
            ctx.num_cols = num_cols

            return V

        @staticmethod
        def backward(ctx, grad_V):
            beta, V, v_init = ctx.saved_tensors
            grad_V_c = grad_V.contiguous()

            K = ctx.K
            num_cols = ctx.num_cols

            grad_beta = torch.empty_like(beta)
            grad_u = torch.empty_like(beta)
            grad_v_init = torch.empty_like(v_init)

            BLOCK = _TritonLinearRecurrence._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _bwd_recurrence_kernel[grid](
                beta, V, v_init, grad_V_c,
                grad_beta, grad_u, grad_v_init,
                K, num_cols,
                BLOCK=BLOCK,
            )

            return grad_beta, grad_u, grad_v_init

    # ============================================================
    # Fused PLIF forward/backward kernels
    # ============================================================

    @triton.jit
    def _fused_plif_fwd_kernel(
        BETA_ptr, U_ptr, VTH_ptr, INIT_ptr,
        OUTPUT_ptr, VPOST_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Bio-inspired PLIF forward: super-threshold current output (ReLU-like).

        NEW semantics (v3, 2026-04-22):
          v = v_init
          for k = 0..K-1:
            v_pre[k] = beta[k]*v + u[k]
            output[k] = max(v_pre[k] - v_th[k], 0)   # super-threshold ion current
            v = v_pre[k] - output[k]                  # = min(v_pre[k], v_th[k])

        Biological interpretation:
          - output[k] is the current (ions) flowing out when v_pre exceeds threshold.
          - Reset: membrane clamps to v_th (after complete discharge to equilibrium).
          - output is naturally SPARSE (= 0 when v_pre <= v_th), enabling sparse GEMM
            downstream. ReLU gradient is exact (no surrogate approximation needed).
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v + u
            output = tl.maximum(v_pre - vth, 0.0)   # super-threshold current (ReLU)
            v = v_pre - output                       # = min(v_pre, vth) (clamp)

            tl.store(OUTPUT_ptr + off, output, mask=mask)
            tl.store(VPOST_ptr + off, v, mask=mask)

    @triton.jit
    def _fused_plif_bwd_kernel(
        BETA_ptr, VTH_ptr, INIT_ptr, VPOST_ptr, OUTPUT_ptr,
        GRAD_OUTPUT_ptr, GRAD_VPOST_ptr,
        GRAD_BETA_ptr, GRAD_U_ptr, GRAD_VTH_ptr, GRAD_INIT_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Bio-inspired PLIF backward: exact ReLU gradient (no surrogate).

        Forward:
          v_pre[k]   = beta[k] * v_post[k-1] + u[k]
          s[k]       = 1 if v_pre[k] > v_th[k] else 0    (fire indicator)
          output[k]  = max(v_pre[k] - v_th[k], 0) = (v_pre[k] - v_th[k]) · s[k]
          v_post[k]  = v_pre[k] - output[k] = v_pre[k]·(1-s[k]) + v_th[k]·s[k]

        Partial derivatives:
          ∂output/∂v_pre = s       ∂output/∂v_th = -s
          ∂v_post/∂v_pre = 1 - s   ∂v_post/∂v_th = s

        Reverse accumulation:
          acc = 0
          for k = K-1 downto 0:
            s = (output[k] > 0) ? 1 : 0
            total_gV = grad_v_post[k] + acc
            grad_v_pre = grad_output[k] * s + total_gV * (1 - s)
            grad_u[k]  = grad_v_pre
            grad_beta[k] = grad_v_pre * v_post[k-1]
            grad_v_th[k] = s * (total_gV - grad_output[k])
            acc = grad_v_pre * beta[k]
          grad_v_init = acc
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        acc = tl.zeros([BLOCK], dtype=tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            output = tl.load(OUTPUT_ptr + off, mask=mask, other=0.0).to(tl.float32)

            g_out = tl.load(GRAD_OUTPUT_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g_V = tl.load(GRAD_VPOST_ptr + off, mask=mask, other=0.0).to(tl.float32)

            # v_post[k-1]
            if k > 0:
                v_prev = tl.load(
                    VPOST_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
            else:
                v_prev = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

            # Exact ReLU gradient: s = 1 if fired (output > 0) else 0
            s = tl.where(output > 0.0, 1.0, 0.0)

            total_gV = g_V + acc
            grad_v_pre = g_out * s + total_gV * (1.0 - s)

            tl.store(GRAD_BETA_ptr + off, grad_v_pre * v_prev, mask=mask)
            tl.store(GRAD_U_ptr + off, grad_v_pre, mask=mask)
            tl.store(GRAD_VTH_ptr + off, s * (total_gV - g_out), mask=mask)

            acc = grad_v_pre * beta

        tl.store(GRAD_INIT_ptr + cols, acc, mask=mask)

    # ============================================================
    # Fused PLIF kernels with row-parameter beta/v_th
    # (constant across K steps — e.g., ParametricLIFNode scalars)
    # ============================================================

    @triton.jit
    def _fused_plif_fwd_rowparam_kernel(
        BETA_ROW_ptr, U_ptr, VTH_ROW_ptr, INIT_ptr,
        OUTPUT_ptr, VPOST_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Bio-inspired PLIF forward (row-parameter β/v_th, constant across K).

        Same math as _fused_plif_fwd_kernel but β/v_th are row-constant, loaded
        once into registers. Reduces HBM reads from 3 per step to 1 (u only).

          output[k] = max(beta*v + u[k] - v_th, 0)
          v         = min(beta*v + u[k], v_th)
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(BETA_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vth = tl.load(VTH_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v + u
            output = tl.maximum(v_pre - vth, 0.0)
            v = v_pre - output                       # = min(v_pre, vth)

            tl.store(OUTPUT_ptr + off, output, mask=mask)
            tl.store(VPOST_ptr + off, v, mask=mask)

    @triton.jit
    def _fused_plif_bwd_rowparam_kernel(
        BETA_ROW_ptr, VTH_ROW_ptr, INIT_ptr, VPOST_ptr, OUTPUT_ptr,
        GRAD_OUTPUT_ptr, GRAD_VPOST_ptr,
        GRAD_BETA_ROW_ptr, GRAD_U_ptr, GRAD_VTH_ROW_ptr, GRAD_INIT_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Bio-inspired PLIF backward (row-parameter, exact ReLU gradient).

        beta/v_th accumulate gradients over K steps in registers.
        Recovery: s = (output > 0), no surrogate needed.
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        beta = tl.load(BETA_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK], dtype=tl.float32)
        acc_grad_beta = tl.zeros([BLOCK], dtype=tl.float32)
        acc_grad_vth = tl.zeros([BLOCK], dtype=tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            output = tl.load(OUTPUT_ptr + off, mask=mask, other=0.0).to(tl.float32)

            g_out = tl.load(GRAD_OUTPUT_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g_V = tl.load(GRAD_VPOST_ptr + off, mask=mask, other=0.0).to(tl.float32)

            if k > 0:
                v_prev = tl.load(
                    VPOST_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
            else:
                v_prev = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

            # Exact ReLU gradient
            s = tl.where(output > 0.0, 1.0, 0.0)

            total_gV = g_V + acc
            grad_v_pre = g_out * s + total_gV * (1.0 - s)

            tl.store(GRAD_U_ptr + off, grad_v_pre, mask=mask)

            acc_grad_beta += grad_v_pre * v_prev
            acc_grad_vth += s * (total_gV - g_out)

            acc = grad_v_pre * beta

        tl.store(GRAD_INIT_ptr + cols, acc, mask=mask)
        tl.store(GRAD_BETA_ROW_ptr + cols, acc_grad_beta, mask=mask)
        tl.store(GRAD_VTH_ROW_ptr + cols, acc_grad_vth, mask=mask)

    class _TritonPLIFRowParamForward(torch.autograd.Function):
        """Fused Triton PLIF with row-parameter beta/v_th.

        For neurons with constant beta/v_th across K steps (ParametricLIFNode).
        Eliminates expand+contiguous for beta/v_th tensors, reduces memory I/O by ~40%.
        """

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta_row, u, v_th_row, v_init):
            beta_row_c = beta_row.contiguous()
            u_c = u.contiguous()
            v_th_row_c = v_th_row.contiguous()
            v_init_c = v_init.contiguous()

            K = u_c.shape[0]
            num_cols = u_c[0].numel()

            output = torch.empty_like(u_c)
            V_post = torch.empty_like(u_c)

            BLOCK = _TritonPLIFRowParamForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_rowparam_kernel[grid](
                beta_row_c, u_c, v_th_row_c, v_init_c,
                output, V_post,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if any(ctx.needs_input_grad[:4]):
                ctx.save_for_backward(beta_row_c, v_th_row_c, v_init_c, V_post, output)
            ctx.K = K
            ctx.num_cols = num_cols

            return output, V_post

        @staticmethod
        def backward(ctx, grad_output, grad_V_post):
            beta_row, v_th_row, v_init, V_post, output = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols

            if grad_output is None:
                grad_output = torch.zeros_like(output)
            if grad_V_post is None:
                grad_V_post = torch.zeros_like(V_post)

            grad_output_c = grad_output.contiguous()
            grad_V_post_c = grad_V_post.contiguous()

            grad_beta_row = torch.empty_like(beta_row)
            grad_u = torch.empty_like(V_post)
            grad_v_th_row = torch.empty_like(v_th_row)
            grad_v_init = torch.empty_like(v_init)

            BLOCK = _TritonPLIFRowParamForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_bwd_rowparam_kernel[grid](
                beta_row, v_th_row, v_init, V_post, output,
                grad_output_c, grad_V_post_c,
                grad_beta_row, grad_u, grad_v_th_row, grad_v_init,
                K, num_cols,
                BLOCK=BLOCK,
            )

            return grad_beta_row, grad_u, grad_v_th_row, grad_v_init

    class _TritonPLIFForward(torch.autograd.Function):
        """Fused Triton PLIF forward + backward.

        Single-pass sequential scan replaces the 3-phase approach:
          Phase 1 (linear scan) + Phase 2 (spike iteration) + Phase 3 (correction)
          → 1 fused kernel with inline spike detection + soft reset

        Advantages:
          - 1 kernel launch (vs 3-4 launches + ~10 element-wise ops)
          - Exact computation (no iteration convergence issues)
          - Less memory (no intermediate V_L, delta_S, delta_S_prev)
          - Higher precision (fp32 accumulation, no bf16 intermediate store/load)
        """

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta, u, v_th, v_init):
            beta_c = beta.contiguous()
            u_c = u.contiguous()
            v_th_c = v_th.contiguous()
            v_init_c = v_init.contiguous()

            K = beta_c.shape[0]
            num_cols = beta_c[0].numel()

            output = torch.empty_like(u_c)
            V_post = torch.empty_like(u_c)

            BLOCK = _TritonPLIFForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_kernel[grid](
                beta_c, u_c, v_th_c, v_init_c,
                output, V_post,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if any(ctx.needs_input_grad[:4]):
                ctx.save_for_backward(beta_c, v_th_c, v_init_c, V_post, output)
            ctx.K = K
            ctx.num_cols = num_cols

            return output, V_post

        @staticmethod
        def backward(ctx, grad_output, grad_V_post):
            beta, v_th, v_init, V_post, output = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols

            if grad_output is None:
                grad_output = torch.zeros_like(output)
            if grad_V_post is None:
                grad_V_post = torch.zeros_like(V_post)

            grad_output_c = grad_output.contiguous()
            grad_V_post_c = grad_V_post.contiguous()

            grad_beta = torch.empty_like(beta)
            grad_u = torch.empty_like(beta)
            grad_v_th = torch.empty_like(v_th)
            grad_v_init = torch.empty_like(v_init)

            BLOCK = _TritonPLIFForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_bwd_kernel[grid](
                beta, v_th, v_init, V_post, output,
                grad_output_c, grad_V_post_c,
                grad_beta, grad_u, grad_v_th, grad_v_init,
                K, num_cols,
                BLOCK=BLOCK,
            )

            return grad_beta, grad_u, grad_v_th, grad_v_init


    # ============================================================
    # V2: V_post-less PLIF variants (reconstruct V_post in backward)
    #
    # Motivation: V_post is (K, seq*batch, DN) — at N=16 K=12 seq=2048 bs=1 that's
    # ~800 MB. In the current impl V_post stays live across the whole training
    # step because `self.hidden_neuron.v = V_post[-1].detach()` holds a view into
    # its storage. V2 returns v_last as a fresh allocation (clone), so V_post can
    # be freed after forward, then reconstructed as scratch in backward via the
    # existing forward kernel. Trade: +1 extra forward pass of PLIF in backward.
    #
    # Saved tensor swap: [beta, v_th, v_init, V_post(big), output(big)]
    #                 → [beta, v_th, v_init, u(big),     output(big)]
    # If u is already saved upstream (e.g. by fused_modulation's autograd), this
    # is a net win. If not, net zero. Run scripts/test_plif_kernel_v2.py to
    # measure empirically.
    # ============================================================

    class _TritonPLIFRowParamForward_V2(torch.autograd.Function):
        """V2: only returns v_last (single step), not full V_post. Recomputes in bwd."""

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta_row, u, v_th_row, v_init):
            beta_row_c = beta_row.contiguous()
            u_c = u.contiguous()
            v_th_row_c = v_th_row.contiguous()
            v_init_c = v_init.contiguous()

            K = u_c.shape[0]
            num_cols = u_c[0].numel()

            output = torch.empty_like(u_c)
            V_post_tmp = torch.empty_like(u_c)   # transient, freed after this function

            BLOCK = _TritonPLIFRowParamForward_V2._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_rowparam_kernel[grid](
                beta_row_c, u_c, v_th_row_c, v_init_c,
                output, V_post_tmp,
                K, num_cols,
                BLOCK=BLOCK,
            )

            # Clone last step into its own allocation; V_post_tmp can then be freed.
            v_last = V_post_tmp[-1].clone()
            # Let V_post_tmp fall out of scope after save_for_backward decision.

            if any(ctx.needs_input_grad[:4]):
                # Save u (needed to recompute V_post in backward). V_post NOT saved.
                ctx.save_for_backward(beta_row_c, v_th_row_c, v_init_c, u_c, output)
            ctx.K = K
            ctx.num_cols = num_cols

            return output, v_last

        @staticmethod
        def backward(ctx, grad_output, grad_v_last):
            beta_row, v_th_row, v_init, u, output = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols

            if grad_output is None:
                grad_output = torch.zeros_like(output)

            # Step 1: recompute V_post as scratch by re-running forward kernel.
            V_post = torch.empty_like(u)
            output_scratch = torch.empty_like(u)  # discarded; output already saved

            BLOCK = _TritonPLIFRowParamForward_V2._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_rowparam_kernel[grid](
                beta_row, u, v_th_row, v_init,
                output_scratch, V_post,
                K, num_cols,
                BLOCK=BLOCK,
            )
            del output_scratch

            # Step 2: propagate grad_v_last into grad_V_post[-1] and run bwd.
            grad_V_post = torch.zeros_like(V_post)
            if grad_v_last is not None:
                grad_V_post[-1] = grad_v_last

            grad_output_c = grad_output.contiguous()
            grad_V_post_c = grad_V_post.contiguous()

            grad_beta_row = torch.empty_like(beta_row)
            grad_u = torch.empty_like(u)
            grad_v_th_row = torch.empty_like(v_th_row)
            grad_v_init = torch.empty_like(v_init)

            _fused_plif_bwd_rowparam_kernel[grid](
                beta_row, v_th_row, v_init, V_post, output,
                grad_output_c, grad_V_post_c,
                grad_beta_row, grad_u, grad_v_th_row, grad_v_init,
                K, num_cols,
                BLOCK=BLOCK,
            )

            return grad_beta_row, grad_u, grad_v_th_row, grad_v_init


    class _TritonPLIFForward_V2(torch.autograd.Function):
        """V2 perstep: only returns v_last. See _TritonPLIFRowParamForward_V2 docs."""

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta, u, v_th, v_init):
            beta_c = beta.contiguous()
            u_c = u.contiguous()
            v_th_c = v_th.contiguous()
            v_init_c = v_init.contiguous()

            K = beta_c.shape[0]
            num_cols = beta_c[0].numel()

            output = torch.empty_like(u_c)
            V_post_tmp = torch.empty_like(u_c)

            BLOCK = _TritonPLIFForward_V2._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_kernel[grid](
                beta_c, u_c, v_th_c, v_init_c,
                output, V_post_tmp,
                K, num_cols,
                BLOCK=BLOCK,
            )

            v_last = V_post_tmp[-1].clone()

            if any(ctx.needs_input_grad[:4]):
                ctx.save_for_backward(beta_c, u_c, v_th_c, v_init_c, output)
            ctx.K = K
            ctx.num_cols = num_cols

            return output, v_last

        @staticmethod
        def backward(ctx, grad_output, grad_v_last):
            beta, u, v_th, v_init, output = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols

            if grad_output is None:
                grad_output = torch.zeros_like(output)

            V_post = torch.empty_like(u)
            output_scratch = torch.empty_like(u)

            BLOCK = _TritonPLIFForward_V2._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_kernel[grid](
                beta, u, v_th, v_init,
                output_scratch, V_post,
                K, num_cols,
                BLOCK=BLOCK,
            )
            del output_scratch

            grad_V_post = torch.zeros_like(V_post)
            if grad_v_last is not None:
                grad_V_post[-1] = grad_v_last

            grad_output_c = grad_output.contiguous()
            grad_V_post_c = grad_V_post.contiguous()

            grad_beta = torch.empty_like(beta)
            grad_u = torch.empty_like(beta)
            grad_v_th = torch.empty_like(v_th)
            grad_v_init = torch.empty_like(v_init)

            _fused_plif_bwd_kernel[grid](
                beta, v_th, v_init, V_post, output,
                grad_output_c, grad_V_post_c,
                grad_beta, grad_u, grad_v_th, grad_v_init,
                K, num_cols,
                BLOCK=BLOCK,
            )

            return grad_beta, grad_u, grad_v_th, grad_v_init


# ============================================================
# Hillis-Steele parallel prefix scan (CPU fallback)
# ============================================================

def hillis_steele_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hillis-Steele 并行前缀扫描：计算仿射映射序列的所有前缀复合。

    给定仿射映射 f_k(x) = a[k] * x + b[k], k = 0, ..., K-1，
    计算前缀复合 F_k = f_k ∘ f_{k-1} ∘ ... ∘ f_0，
    使得 V[k] = F_k(v_init) = A[k] * v_init + B[k]。

    复合规则: (a2, b2) ∘ (a1, b1) = (a2 * a1, a2 * b1 + b2)

    实现使用 torch.cat 重建张量（无原地操作），完全兼容 autograd。

    Args:
        a: (K, *shape) — 乘性系数（如 β）
        b: (K, *shape) — 加性项（如 α·I）

    Returns:
        A: (K, *shape) — 前缀积 A[k] = ∏_{j=0}^{k} a[j]
        B: (K, *shape) — 前缀和 B[k] 使得 V[k] = A[k] * v_init + B[k]

    并行深度: O(log K)
    工作量: O(K * log K)
    """
    K = a.shape[0]
    A = a
    B = b

    d = 1
    while d < K:
        A_new_tail = A[d:] * A[:-d]
        B_new_tail = A[d:] * B[:-d] + B[d:]

        A = torch.cat([A[:d], A_new_tail], dim=0)
        B = torch.cat([B[:d], B_new_tail], dim=0)

        d *= 2

    return A, B

# ============================================================
# Public API: linear_recurrence
# ============================================================

def linear_recurrence(beta: torch.Tensor, u: torch.Tensor, v_init: torch.Tensor) -> torch.Tensor:
    """
    求解线性递推: V[k] = beta[k] * V[k-1] + u[k], V[-1] = v_init

    CUDA 后端: Triton fused kernel（1 次 kernel launch，O(K) 工作量）
    CPU 后端:  Hillis-Steele parallel scan（O(K log K) 工作量）

    Args:
        beta: (K, *shape) — 衰减系数，值域 (0, 1)
        u:    (K, *shape) — 输入项
        v_init: (*shape) — 初始状态

    Returns:
        V: (K, *shape) — 所有 K 步的状态
    """
    if _HAS_TRITON and beta.is_cuda:
        return _TritonLinearRecurrence.apply(beta, u, v_init)
    # CPU fallback
    A, B = hillis_steele_scan(beta, u)
    V = A * v_init.unsqueeze(0) + B
    return V

# ============================================================
# PLIF parallel forward (with spike iteration)
# ============================================================

def plif_parallel_forward(
    beta: torch.Tensor,
    u: torch.Tensor,
    v_th: torch.Tensor,
    v_init: torch.Tensor,
    max_iter: int = 3,
    surrogate_function=None,  # kept for API-compat; ignored in bio-ReLU path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bio-inspired PLIF forward pass (v3, super-threshold current output).

    Computes:
        V_pre[k]  = beta[k] * V_post[k-1] + u[k]
        output[k] = max(V_pre[k] - v_th[k], 0)   # ReLU: super-threshold amount
        V_post[k] = V_pre[k] - output[k]          # = min(V_pre[k], v_th[k])

    Triton path (CUDA): fused kernel, exact ReLU gradient.
    CPU fallback: sequential Python loop (slow but correct).

    Args:
        beta:  (K, *shape) — 衰减系数
        u:     (K, *shape) — 输入 α·I
        v_th:  (K, *shape) — 动态阈值
        v_init: (*shape) — 初始膜电位
        max_iter / surrogate_function: deprecated no-ops (kept for call-site compat)

    Returns:
        output: (K, *shape) — 超阈值电流 (sparse, ReLU)
        V_post: (K, *shape) — 发放后膜电位 (= min(V_pre, v_th))
        V_pre:  always None (no longer reconstructed)
    """
    if _HAS_TRITON and beta.is_cuda:
        output, V_post = _TritonPLIFForward.apply(beta, u, v_th, v_init)
        return output, V_post, None

    # CPU / non-CUDA fallback: sequential loop (correct, slow)
    K = u.shape[0]
    outputs = []
    V_posts = []
    v = v_init if isinstance(v_init, torch.Tensor) else torch.zeros_like(u[0])
    for k in range(K):
        v_pre = beta[k] * v + u[k]
        output = F.relu(v_pre - v_th[k])   # super-threshold current (sparse)
        v = v_pre - output                  # = min(v_pre, v_th[k])
        outputs.append(output)
        V_posts.append(v)
    return torch.stack(outputs, dim=0), torch.stack(V_posts, dim=0), None


def plif_parallel_forward_v2(
    beta: torch.Tensor, u: torch.Tensor, v_th: torch.Tensor, v_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V2: returns (output, v_last) instead of (output, V_post, None).

    v_last is a fresh (not-view) tensor of shape v_init. V_post is freed after
    forward — downstream `neuron.v = v_last.detach()` no longer holds the full
    (K, *) V_post storage across steps.
    """
    if _HAS_TRITON and beta.is_cuda:
        return _TritonPLIFForward_V2.apply(beta, u, v_th, v_init)
    # CPU fallback: reuse old path and clone last step
    K = u.shape[0]
    v = v_init if isinstance(v_init, torch.Tensor) else torch.zeros_like(u[0])
    outputs = []
    for k in range(K):
        v_pre = beta[k] * v + u[k]
        output = F.relu(v_pre - v_th[k])
        v = v_pre - output
        outputs.append(output)
    return torch.stack(outputs, dim=0), v.clone()


def plif_rowparam_forward_v2(
    beta_row: torch.Tensor, u: torch.Tensor, v_th_row: torch.Tensor, v_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V2 rowparam: returns (output, v_last) instead of (output, V_post). See V2 docstring."""
    if _HAS_TRITON and u.is_cuda:
        return _TritonPLIFRowParamForward_V2.apply(beta_row, u, v_th_row, v_init)
    K = u.shape[0]
    v = v_init if isinstance(v_init, torch.Tensor) else torch.zeros_like(u[0])
    outputs = []
    for k in range(K):
        v_pre = beta_row * v + u[k]
        output = F.relu(v_pre - v_th_row)
        v = v_pre - output
        outputs.append(output)
    return torch.stack(outputs, dim=0), v.clone()


def plif_rowparam_forward(
    beta_row: torch.Tensor,
    u: torch.Tensor,
    v_th_row: torch.Tensor,
    v_init: torch.Tensor,
    surrogate_function=None,  # deprecated, kept for API-compat
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bio-inspired PLIF forward (row-parameter, β/v_th constant across K).

    Args:
        beta_row: (*shape) — 每列的衰减率 (所有 K 步相同)
        u:        (K, *shape) — 每步输入 (α·I)
        v_th_row: (*shape) — 每列的阈值 (所有 K 步相同)
        v_init:   (*shape) — 初始膜电位
        surrogate_function: deprecated no-op (kept for call-site compat)

    Returns:
        output: (K, *shape) — 超阈值电流 (sparse)
        V_post: (K, *shape) — 发放后膜电位
    """
    if _HAS_TRITON and u.is_cuda:
        output, V_post = _TritonPLIFRowParamForward.apply(
            beta_row, u, v_th_row, v_init,
        )
        return output, V_post

    # CPU fallback: sequential loop
    K = u.shape[0]
    outputs = []
    V_posts = []
    v = v_init if isinstance(v_init, torch.Tensor) else torch.zeros_like(u[0])
    for k in range(K):
        v_pre = beta_row * v + u[k]
        output = F.relu(v_pre - v_th_row)
        v = v_pre - output
        outputs.append(output)
        V_posts.append(v)
    return torch.stack(outputs, dim=0), torch.stack(V_posts, dim=0)

def plif_fixed_param_forward(
    beta,
    u: torch.Tensor,
    v_th,
    v_init: torch.Tensor,
    max_iter: int = 3,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    固定参数 PLIF 神经元的并行前向（如输出神经元、FFN 神经元）。

    ParametricLIFNode 方程: V[k] = beta * V[k-1] + (1-beta) * x[k]
    其中 beta = 1/(1+exp(w)), 可为 scalar tensor（保持梯度流向 w）。

    scalar/0-dim beta 和 v_th 使用 row-param 内核（无需 expand 到 (K, *shape)）。

    Args:
        beta: 衰减率 — scalar float、0-dim tensor 或 (K, *shape) tensor
        u: (K, *shape) — 输入（已乘以 (1-beta)）
        v_th: 阈值 — scalar float、0-dim tensor 或 (K, *shape) tensor
        v_init: (*shape) — 初始膜电位
        max_iter: spike 迭代次数
        surrogate_function: surrogate gradient 函数

    Returns:
        spike: (K, *shape) — spike 模式
        V_post: (K, *shape) — 发放后膜电位
    """
    K = u.shape[0]
    shape = u.shape[1:]

    # Row-param fast path: beta 和 v_th 都是 scalar/0-dim → 扩展为 (*shape) 行向量
    beta_is_scalar = isinstance(beta, torch.Tensor) and beta.dim() == 0
    beta_is_float = not isinstance(beta, torch.Tensor)
    vth_is_scalar = isinstance(v_th, torch.Tensor) and v_th.dim() == 0
    vth_is_float = not isinstance(v_th, torch.Tensor)

    if (beta_is_scalar or beta_is_float) and (vth_is_scalar or vth_is_float):
        # Build row vectors (*shape)
        if beta_is_scalar:
            beta_row = beta.expand(*shape).contiguous()
        else:
            beta_row = torch.full(shape, beta, device=u.device, dtype=u.dtype)
        if vth_is_scalar:
            v_th_row = v_th.expand(*shape).contiguous()
        else:
            v_th_row = torch.full(shape, v_th, device=u.device, dtype=u.dtype)
        return plif_rowparam_forward(beta_row, u, v_th_row, v_init, surrogate_function)

    # Full-tensor path: expand to (K, *shape) if needed
    if isinstance(beta, torch.Tensor):
        if beta.dim() == 0:
            beta = beta.expand(K, *shape).contiguous()
    else:
        beta = torch.full_like(u, beta)

    if isinstance(v_th, torch.Tensor):
        if v_th.dim() == 0:
            v_th = v_th.expand(K, *shape).contiguous()
    else:
        v_th = torch.full_like(u, v_th)

    spike, V_post, _ = plif_parallel_forward(
        beta, u, v_th, v_init, max_iter, surrogate_function,
    )
    return spike, V_post


# ============================================================
# Section: plif_node
# ============================================================

class PLIFNode(MemoryModule):
    """
    D 维固定参数 PLIF 神经元。

    Args:
        dim: 神经元数量（每个维度独立参数）
        init_tau: 初始时间常数 τ（β = 1 - 1/τ）
        v_threshold: 初始发放阈值
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        dim: int,
        init_tau: float = 2.0,
        v_threshold: float = 0.5,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        # D 维可学习参数（随机初始化，每个维度独立）
        # w: 控制 β=sigmoid(w)，随机产生不同时间常数
        #    init_w ± 0.5 → β ∈ ~[sigmoid(w-0.5), sigmoid(w+0.5)]
        #    tau=2.0 时 w=0, β ∈ ~[0.38, 0.62]
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.empty(dim).normal_(init_w, 0.5))
        # v_th: 发放阈值. bio-ReLU 下 output = max(V_pre - V_th, 0),
        # V_pre 在 init 时近似 N(0, σ²) 对称分布, 最大理论发放率 50% (V_th→0),
        # 为让初始发放率贴近 40-50% 的健康区 (不是 dead-ReLU 区), v_th 初始范围
        # 从旧 [0.5x, 1.5x] 缩小到 [0.05x, 0.3x]×v_threshold (均值约 0.18×).
        # 每维保持多样性 (不同神经元有不同时间尺度 + 不同阈值).
        self.v_th = nn.Parameter(torch.empty(dim).uniform_(
            v_threshold * 0.05, v_threshold * 0.3,
        ))
        self.surrogate_function = surrogate_function
        # 膜电位状态（functional.reset_net 时重置为 0.）
        self.register_memory('v', 0.)

    @property
    def beta(self):
        """D 维衰减率 β = sigmoid(w)，值域 (0, 1)。"""
        return torch.sigmoid(self.w)

    def forward(self, x):
        """
        单步 bio-inspired PLIF 前向:
          V_pre  = β · V_{t-1} + (1-β) · x
          output = max(V_pre - V_th, 0)   # 超阈电流 (ReLU, sparse)
          V_t    = V_pre - output         # = min(V_pre, V_th) (降到阈值)

        Returns:
            output: 连续值 (超阈电流), shape (batch, dim), 非发放位置 == 0
        """
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x)
        beta = self.beta
        v_pre = beta * self.v + (1.0 - beta) * x
        output = F.relu(v_pre - self.v_th)
        self.v = v_pre - output
        return output


# ============================================================
# Section: selective_plif
# ============================================================

class SelectivePLIFNode(BaseNode):
    """
    隐状态空间的核心神经元。

    接收外部动态计算的 β(t), α(t), V_th(t)，执行：
      charge → fire → soft reset

    Args:
        surrogate_function: surrogate gradient 函数，默认 Sigmoid(alpha=4.0)
        detach_reset: 是否在 reset 时 detach spike，默认 False
    """

    def __init__(
        self,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        detach_reset: bool = False,
    ):
        # v_threshold=1.0 是占位值，实际使用外部传入的 v_th
        # v_reset=None 触发 soft reset 模式，register_memory('v', 0.)
        super().__init__(
            v_threshold=1.0,
            v_reset=None,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode='s',
            backend='torch',
            store_v_seq=False,
        )

    def single_step_forward(
        self,
        x: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        v_th: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bio-inspired 单步前向 (v3 超阈电流):
          V_pre  = β(t) · V[t-1] + α(t) · I[t]
          output = max(V_pre - V_th(t), 0)    # 超阈电流 (ReLU, sparse)
          V[t]   = V_pre - output             # = min(V_pre, V_th(t))

        Args:
            x:    输入电流 I[t],   shape (batch, D*N)
            beta: 衰减率 β(t),    shape (batch, D*N), 值域 (0, 1)
            alpha: 写入增益 α(t),  shape (batch, D*N), 值域 R+
            v_th: 动态阈值 V_th(t), shape (batch, D*N), 值域 R+

        Returns:
            output: 超阈电流 (连续, sparse), shape (batch, D*N), 非发放位置 == 0
        """
        self.v_float_to_tensor(x)

        v_pre = beta * self.v + alpha * x
        output = F.relu(v_pre - v_th)
        self.v = v_pre - output
        return output

    def extra_repr(self) -> str:
        return (
            f'v_reset={self.v_reset}, '
            f'detach_reset={self.detach_reset}, '
            f'step_mode={self.step_mode}, '
            f'surrogate={self.surrogate_function}'
        )


# ============================================================
# Section: lateral_inhibition
# ============================================================

# ============================================================
# Triton fused kernels
# ============================================================

if _HAS_TRITON:

    @triton.jit
    def _li_fwd_kernel(
        X_ptr, GAIN_ptr, OUT_ptr,
        stride_row,
        D: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Forward: out = x * rsqrt(mean(x²) + eps) * gain

        Grid: (num_rows,). Each program processes one row of D elements.
        Computation in float32; storage in input dtype.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        off = row * stride_row + cols

        # Load in float32
        x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
        gain = tl.load(GAIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Inhibitory pool: population activity
        variance = tl.sum(x * x, axis=0) / D
        rrms = 1.0 / tl.sqrt(variance + eps)

        # Divisive inhibition + gain modulation
        out = x * rrms * gain

        tl.store(OUT_ptr + off, out, mask=mask)

    @triton.jit
    def _li_bwd_kernel(
        DOUT_ptr, X_ptr, GAIN_ptr,
        DX_ptr, DGAIN_ptr,
        stride_row,
        D: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Backward: grad_x, grad_gain (per-row, reduced externally).

        Grid: (num_rows,).
        d_x = rrms * (d_out * gain - x_hat * mean(d_out * gain * x_hat))
        d_gain_row = d_out * x_hat  (sum across rows done outside kernel)
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        off = row * stride_row + cols

        dout = tl.load(DOUT_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
        gain = tl.load(GAIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Recompute forward (avoid saving intermediate tensors)
        variance = tl.sum(x * x, axis=0) / D
        rrms = 1.0 / tl.sqrt(variance + eps)
        x_hat = x * rrms

        # grad_gain (per-row contribution)
        dgain = dout * x_hat
        tl.store(DGAIN_ptr + off, dgain, mask=mask)

        # grad_x: rrms * (dout*gain - x_hat * mean(dout*gain*x_hat))
        dout_gain = dout * gain
        dot = tl.sum(dout_gain * x_hat, axis=0) / D
        dx = (dout_gain - x_hat * dot) * rrms

        tl.store(DX_ptr + off, dx, mask=mask)

class _LateralInhibitionTriton(torch.autograd.Function):
    """Triton-accelerated lateral inhibition (divisive normalization)."""

    @staticmethod
    def forward(ctx, x, gain, eps):
        orig_shape = x.shape
        D = x.shape[-1]
        x_2d = x.reshape(-1, D).contiguous()
        N = x_2d.shape[0]

        out = torch.empty_like(x_2d)

        BLOCK_D = triton.next_power_of_2(D)
        _li_fwd_kernel[(N,)](
            x_2d, gain, out,
            x_2d.stride(0),
            D=D, eps=eps, BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_2d, gain)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.N = N
        ctx.D = D

        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, gain = ctx.saved_tensors
        D = ctx.D
        N = ctx.N

        grad_2d = grad_output.reshape(N, D).contiguous()

        dx = torch.empty_like(x_2d)
        dgain_rows = torch.empty_like(x_2d)

        BLOCK_D = triton.next_power_of_2(D)
        _li_bwd_kernel[(N,)](
            grad_2d, x_2d, gain,
            dx, dgain_rows,
            x_2d.stride(0),
            D=D, eps=ctx.eps, BLOCK_D=BLOCK_D,
        )

        # Reduce per-row dgain across all rows
        dgain = dgain_rows.sum(dim=0)

        return dx.reshape(ctx.orig_shape), dgain, None

# ============================================================
# Public module
# ============================================================

class LateralInhibition(MemoryModule):
    """
    侧抑制归一化层（Divisive Normalization）。

    通过抑制性中间神经元池实现增益控制。

    数学：
      pool = mean(h², dim=-1)            # 抑制性池：群体活动水平
      h_norm = h / sqrt(pool + ε)        # 分裂抑制 (shunting inhibition)
      output = gain · h_norm             # 增益调制 (gain modulation)

    等价于 RMSNorm，但在 SNN 框架中对应 divisive normalization
    (Carandini & Heeger, 2012)，是神经科学中最基本的计算原语之一。

    CUDA: Triton fused kernel（前向+反向各 1 次 launch）
    CPU:  PyTorch fallback

    Args:
        dim: 特征维度（D）
        eps: 数值稳定性
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if _HAS_TRITON and h.is_cuda:
            return _LateralInhibitionTriton.apply(h, self.gain, self.eps)
        # PyTorch fallback
        variance = h.pow(2).mean(-1, keepdim=True)
        h_norm = h * torch.rsqrt(variance + self.eps)
        return self.gain * h_norm

    def extra_repr(self):
        return f'dim={self.dim}, eps={self.eps}'


# ============================================================
# Section: snn_ffn
# ============================================================

class SNNFFN(MemoryModule):
    """
    SNN 等价的 Feed-Forward Network。

    Args:
        D: 可见维度（输入/输出 activation 维度）
        D_ff: 中间层维度（对标 Qwen3 intermediate_size）
        output_v_threshold: 输出神经元阈值
        num_layers: 总层数，用于 down_proj 缩放
        layer_idx: 当前层索引
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        D: int,
        D_ff: int,
        output_v_threshold: float = 0.3,
        num_layers: int = 1,
        layer_idx: int = 0,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.D = D
        self.D_ff = D_ff

        # ====== 三条投影路径（对标 SwiGLU: gate_proj, up_proj, down_proj） ======
        self.gate_proj = layer.Linear(D, D_ff, bias=False, step_mode='s')
        self.up_proj = layer.Linear(D, D_ff, bias=False, step_mode='s')
        self.down_proj = layer.Linear(D_ff, D, bias=False, step_mode='s')

        # ====== 残差路径 ======
        self.skip_proj = layer.Linear(D, D, bias=False, step_mode='s')

        # ====== 神经元（D 维或 D_ff 维可学习 β 和 V_th） ======
        # gate_neuron: 门控发放
        self.gate_neuron = PLIFNode(
            dim=D_ff,
            init_tau=2.0,
            v_threshold=output_v_threshold,
            surrogate_function=surrogate_function,
        )
        # up_neuron: 值发放
        self.up_neuron = PLIFNode(
            dim=D_ff,
            init_tau=2.0,
            v_threshold=output_v_threshold,
            surrogate_function=surrogate_function,
        )
        # ====== 参数初始化 ======
        self._initialize_parameters(num_layers)

    def _initialize_parameters(self, num_layers: int):
        """初始化投影权重。

        - gate_proj, up_proj, skip_proj: Kaiming uniform
        - down_proj: Kaiming uniform × 1/√(num_layers)，防深层梯度爆炸
        """
        for lin in [self.gate_proj, self.up_proj, self.skip_proj]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        self.down_proj.weight.data.mul_(1.0 / math.sqrt(num_layers))

    def forward_parallel(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        并行前向传播：使用 parallel scan 处理全序列。

        优化：
          - gate_proj + up_proj 合并为单次 matmul（2 launch → 1）
          - gate + up PLIF scan: row-param kernel（无需 expand+contiguous beta/v_th）
          - u_merged: 向量缩放替代 cat（1次 broadcast multiply 替代 2次 scale + 1次 cat）

        Args:
            h_seq: (TK, batch, D) — 全部 T×K 帧的连续激活（来自 PLIFNode V_post 泄漏量）

        Returns:
            continuous_out: (TK, batch, D) — 全部 T×K 帧的连续输出
        """
        TK, batch, D = h_seq.shape
        input_dtype = h_seq.dtype
        D_ff = self.D_ff
        flat = h_seq.reshape(TK * batch, D)

        # ====== Phase 1: 批量投影（gate+up 合并为 1 次 matmul） ======
        W_gate_up = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        I_gate_up = F.linear(flat, W_gate_up).reshape(TK, batch, 2 * D_ff)
        I_skip = F.linear(flat, self.skip_proj.weight).reshape(TK, batch, D)

        # ====== Phase 2: Gate+Up 合并 PLIF scan（row-param kernel） ======
        beta_gate = self.gate_neuron.beta.to(input_dtype)  # (D_ff,)
        beta_up = self.up_neuron.beta.to(input_dtype)      # (D_ff,)
        surr = self.gate_neuron.surrogate_function

        # u_merged: 向量缩放（D_ff 维 β 直接 cat，无需 expand）
        scale_row = torch.cat([1.0 - beta_gate, 1.0 - beta_up])  # (2*D_ff,)
        u_merged = I_gate_up * scale_row  # (TK, batch, 2*D_ff), broadcast

        # beta_row / v_th_row: (batch, 2*D_ff) — D_ff 维可学习参数
        beta_row = torch.cat([beta_gate, beta_up])  # (2*D_ff,)
        beta_row = beta_row.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()

        v_th_row = torch.cat([self.gate_neuron.v_th.to(input_dtype),
                              self.up_neuron.v_th.to(input_dtype)])  # (2*D_ff,)
        v_th_row = v_th_row.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()

        # v_init_merged: (batch, 2*D_ff)
        v_init_gate = self.gate_neuron.v
        if isinstance(v_init_gate, float):
            v_init_gate = torch.zeros(batch, D_ff, device=flat.device, dtype=flat.dtype)
        v_init_up = self.up_neuron.v
        if isinstance(v_init_up, float):
            v_init_up = torch.zeros(batch, D_ff, device=flat.device, dtype=flat.dtype)
        v_init_merged = torch.cat([v_init_gate, v_init_up], dim=-1)

        # Row-param bio-PLIF scan (V2): returns (output, v_last) not full V_post.
        # v_last is a fresh tensor — full V_post is freed after forward.
        output_merged, v_last_merged = plif_rowparam_forward_v2(
            beta_row, u_merged, v_th_row, v_init_merged,
        )

        # 超阈电流输出 (sparse): 非发放位置严格 == 0, 发放位置 = V_pre - V_th > 0
        gate_out = output_merged[:, :, :D_ff]
        up_out = output_merged[:, :, D_ff:]
        self.gate_neuron.v = v_last_merged[:, :D_ff].detach()
        self.up_neuron.v = v_last_merged[:, D_ff:].detach()
        with torch.no_grad():
            self.gate_neuron._last_firing_rate = (gate_out > 0).float().mean().item()
            self.up_neuron._last_firing_rate = (up_out > 0).float().mean().item()

        # ====== Phase 3: 连续门控（对标 SwiGLU）+ 降维 ======
        gated = gate_out * up_out  # (TK, batch, D_ff), sparser than gate_out alone
        gated_flat = gated.reshape(TK * batch, D_ff)
        I_out = F.linear(gated_flat, self.down_proj.weight).reshape(TK, batch, D) + I_skip

        # 子层级 output_neuron 已移除，改由层级 K 帧聚合处理（模型级 output_neuron 仍保留于 model.py）
        return I_out.to(input_dtype)  # (TK, batch, D), 连续值

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单步前向（推理 autoregressive 模式）。与 forward_parallel 等价:
          output = max(β·v + (1-β)·proj(x) - v_th, 0)
        """
        # 门控路径: 超阈电流输出
        gate_u = (1.0 - self.gate_neuron.beta) * self.gate_proj(x)
        gate_v_pre = self.gate_neuron.beta * self.gate_neuron.v + gate_u
        gate_out = F.relu(gate_v_pre - self.gate_neuron.v_th)
        self.gate_neuron.v = gate_v_pre - gate_out  # = min(v_pre, v_th)

        # 值路径
        up_u = (1.0 - self.up_neuron.beta) * self.up_proj(x)
        up_v_pre = self.up_neuron.beta * self.up_neuron.v + up_u
        up_out = F.relu(up_v_pre - self.up_neuron.v_th)
        self.up_neuron.v = up_v_pre - up_out

        # SwiGLU 风格门控 + down
        gated = gate_out * up_out
        I_out = self.down_proj(gated) + self.skip_proj(x)
        return I_out


# ============================================================
# Section: snn_block
# ============================================================

# ====== Fused modulation activations (torch.compile) ======
# Fuse sigmoid + softplus + abs + alpha*I into single kernel.
# 7-8 separate element-wise kernels → 1 fused kernel, ~4x speedup on DN-sized tensors.
# First call triggers JIT compilation (~seconds); cached for subsequent calls.

def _fused_modulation_eager(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all):
    """Plain-Python modulation: sigmoid + softplus + abs + multiply.
    Used directly for A/B test against compiled version."""
    beta = torch.sigmoid(raw_beta + b_beta)
    alpha = F.softplus(raw_alpha + b_alpha)
    v_th = v_th_min + torch.abs(raw_th + b_th)
    u = alpha * I_all
    return beta, u, v_th


# Default: use compiled version (module-level so bench can swap to eager for A/B)
_fused_modulation = torch.compile(_fused_modulation_eager, backend='inductor', fullgraph=True)


# Module-level flag: whether to activation-checkpoint _fused_modulation.
# EMPIRICALLY DEFAULT FALSE: bench shows checkpoint + torch.compile combo
# INCREASES peak memory (2.96 → 4.67 GB on small config), likely due to
# re-tracing overhead + nested checkpointing with outer layer ckpt.
# Kept as a flag for future experiments (e.g., if layer ckpt is removed).
_FUSED_MODULATION_CHECKPOINT = False

class SNNBlock(MemoryModule):
    """
    单个 SNN Block（并行化）。

    Args:
        D: 可见维度（Block 间通信的维度）
        N: 状态扩展因子（每个通道的隐神经元数）
        v_th_min: 动态阈值下限
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        D: int,
        N: int = 8,
        v_th_min: float = 0.1,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.D = D
        self.N = N
        self.v_th_min = v_th_min
        DN = D * N

        # ====== 因果卷积：局部上下文预混合（参考 Qwen3.5 GatedDeltaNet） ======
        self.conv_kernel_size = 4
        self.conv1d = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=self.conv_kernel_size,
            groups=D,  # 深度卷积
            padding=self.conv_kernel_size - 1,  # causal padding
            bias=False,
        )

        # ====== 六条并行输入投影（SNN 突触：连续激活输入） ======
        self.W_in = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_beta_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_alpha_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_th_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_gate = layer.Linear(D, D, bias=False, step_mode='s')
        self.W_skip = layer.Linear(D, D, bias=False, step_mode='s')

        # ====== β/α/V_th 仅依赖 x（无 W^(V)·V 项） ======

        # ====== 调制偏置（结构化初始化） ======
        self.b_beta = nn.Parameter(torch.empty(DN))
        self.b_alpha = nn.Parameter(torch.empty(DN))
        self.b_th = nn.Parameter(torch.empty(DN))

        # ====== 输出投影：D*N → D（SNN 突触） ======
        self.W_out = layer.Linear(DN, D, bias=False, step_mode='s')

        # ====== 隐状态空间神经元（D*N 个，动态参数） ======
        self.hidden_neuron = SelectivePLIFNode(
            surrogate_function=surrogate_function,
            detach_reset=False,
        )

        # ====== 参数初始化 ======
        self._initialize_parameters()

    def _initialize_parameters(self):
        """功能引导初始化。"""
        D, N = self.D, self.N
        K_ref = 16

        # 目标 β 分布：多时间尺度 [0.80, 0.99]
        beta_values = torch.linspace(0.80, 0.99, N)

        # ====== 1. β 偏置：logit-spaced + 维度间随机扰动 ======
        b_beta_per_n = torch.log(beta_values / (1.0 - beta_values))
        # 以 per_n 值为均值，加 N(0, 0.1) 扰动打破 D 个通道的对称性
        self.b_beta.data.copy_(b_beta_per_n.repeat(D))
        self.b_beta.data.add_(torch.empty_like(self.b_beta).normal_(0, 0.1))

        # ====== 2. α 偏置：softplus(0.5413) ≈ 1.0 + 维度间随机扰动 ======
        # 以 0.5413 为均值，N(0, 0.1) 扰动 → α ∈ ~[0.7, 1.3]
        self.b_alpha.data.normal_(0.5413, 0.1)

        # ====== 3. W^(x) 权重 ======
        for lin in [self.W_in, self.W_gate, self.W_skip, self.W_out]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        for lin in [self.W_beta_x, self.W_alpha_x, self.W_th_x]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            lin.weight.data.mul_(0.1)

        # ====== 4. W_in 时间尺度缩放 ======
        scale_per_n = torch.sqrt(1.0 - beta_values ** 2)  # (N,)
        scale_DN = scale_per_n.repeat(D)  # (D*N,)
        with torch.no_grad():
            self.W_in.weight.mul_(scale_DN.unsqueeze(1))

        # ====== 5. b_th：σ_V 校准 ======
        # σ_V = sqrt(p/3) * sqrt(1 - β^{2K})
        # 其中 p 是输入 firing rate。旧版假设 p=0.5（σ_I=0.408），
        # 但实际 input_neuron firing rate 约 0.07~0.45，深层更低。
        # 用 p=0.15 保守估计，避免 v_th 过高导致死神经元。
        p_assumed = 0.15
        sigma_I_base = math.sqrt(p_assumed / 3.0)
        sigma_V_per_n = sigma_I_base * torch.sqrt(
            1.0 - beta_values ** (2 * K_ref)
        )
        target_p_fire = torch.linspace(0.25, 0.08, N)
        z_scores = math.sqrt(2.0) * torch.erfinv(
            2.0 * (1.0 - target_p_fire) - 1.0
        )
        target_V_th = sigma_V_per_n * z_scores
        b_th_per_n = torch.clamp(target_V_th - self.v_th_min, min=0.05)
        # 以 per_n 值为均值，加 N(0, 0.02) 扰动打破 D 个通道的对称性
        self.b_th.data.copy_(b_th_per_n.repeat(D))
        self.b_th.data.add_(torch.empty_like(self.b_th).normal_(0, 0.02))

        # ====== 6. W_out 发放率均衡缩放 ======
        out_scale_per_n = 1.0 / torch.sqrt(target_p_fire)
        out_scale_per_n = out_scale_per_n / out_scale_per_n.mean()
        out_scale_DN = out_scale_per_n.repeat(D)
        with torch.no_grad():
            self.W_out.weight.mul_(out_scale_DN.unsqueeze(0))

    def forward_parallel(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        并行前向传播：使用 parallel scan 处理全序列。

        Args:
            h_seq: (TK, batch, D) — 全部 T×K 帧的连续激活（来自 PLIFNode V_post 泄漏量）

        Returns:
            continuous_out: (TK, batch, D) — 全部 T×K 帧的连续输出（V_post 经 W_out 投影）
        """
        TK, batch, D = h_seq.shape
        DN = self.D * self.N

        # ====== Phase 0: 因果卷积预混合（局部上下文） ======
        conv_dtype = self.conv1d.weight.dtype
        conv_in = h_seq.to(conv_dtype).permute(1, 2, 0)  # (batch, D, TK)
        conv_out = self.conv1d(conv_in)[:, :, :TK]  # causal: 截断未来
        h_seq = conv_out.permute(2, 0, 1)  # (TK, batch, D)

        # ====== Phase 1: 批量投影（全部 TK 帧同时计算）======
        flat = h_seq.reshape(TK * batch, D)

        I_all = F.linear(flat, self.W_in.weight).reshape(TK, batch, DN)
        raw_beta = F.linear(flat, self.W_beta_x.weight).reshape(TK, batch, DN)
        raw_alpha = F.linear(flat, self.W_alpha_x.weight).reshape(TK, batch, DN)
        raw_th = F.linear(flat, self.W_th_x.weight).reshape(TK, batch, DN)
        gate_all = torch.sigmoid(
            F.linear(flat, self.W_gate.weight).reshape(TK, batch, D)
        )
        I_skip_all = F.linear(flat, self.W_skip.weight).reshape(TK, batch, D)

        # ====== Phase 1b: 融合激活（torch.compile → 单 kernel）======
        # 神经元参数可能是 fp32 (master weights)，计算时转为输入 dtype
        compute_dtype = h_seq.dtype
        # checkpoint: don't save internal intermediates (alpha, etc.). Re-run
        # modulation forward during backward. Saves ~3-5 GB/layer @ N=16.
        # use_reentrant=False enables graph re-tracing under torch.compile.
        if _FUSED_MODULATION_CHECKPOINT and self.training and I_all.requires_grad:
            beta_all, u_hidden, v_th_all = checkpoint(
                _fused_modulation,
                raw_beta, self.b_beta.to(compute_dtype),
                raw_alpha, self.b_alpha.to(compute_dtype),
                raw_th, self.b_th.to(compute_dtype),
                self.v_th_min, I_all,
                use_reentrant=False,
            )
        else:
            beta_all, u_hidden, v_th_all = _fused_modulation(
                raw_beta, self.b_beta.to(compute_dtype),
                raw_alpha, self.b_alpha.to(compute_dtype),
                raw_th, self.b_th.to(compute_dtype),
                self.v_th_min, I_all,
            )

        # 获取隐神经元初始状态
        v_init_hidden = self.hidden_neuron.v
        if isinstance(v_init_hidden, float):
            v_init_hidden = torch.zeros(batch, DN, device=flat.device, dtype=flat.dtype)

        # V2: returns (output, v_last) — V_post not held across steps by .v view
        output_hidden, v_last_hidden = plif_parallel_forward_v2(
            beta_all, u_hidden, v_th_all, v_init_hidden,
        )

        # 更新隐神经元状态（保存末步 V_post 供下次调用）
        self.hidden_neuron.v = v_last_hidden.detach()
        with torch.no_grad():
            self.hidden_neuron._last_firing_rate = (output_hidden > 0).float().mean().item()

        # ====== Phase 4: 输出投影（超阈电流 → W_out: ReLU 精确梯度 + sparse GEMM 潜力）======
        # output_hidden = max(V_pre - v_th, 0) 是 bio-inspired 超阈电流输出:
        #   - 发放时 output = V_pre - V_th > 0, 不发放时 output == 0 (严格稀疏)
        #   - 生物上对应"离子流过突触"的量, W_out 只对"真正发放"的神经元做 GEMM
        #   - 梯度 ∂output/∂V_pre = s (exact ReLU, 不需 surrogate)
        out_flat = output_hidden.reshape(TK * batch, DN)
        I_out_all = F.linear(out_flat, self.W_out.weight).reshape(TK, batch, D)
        I_total_all = I_out_all * gate_all + I_skip_all  # (TK, batch, D)

        # 子层级 output_neuron 已移除，改由层级 K 帧聚合处理（模型级 output_neuron 仍保留于 model.py）
        return I_total_all  # (TK, batch, D), 连续值

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播（用于调试/兼容）。

        Args:
            x: 连续激活输入, shape (batch, D)

        Returns:
            continuous_out: 连续输出, shape (batch, D)
        """
        V_prev = self.hidden_neuron.v
        if isinstance(V_prev, float):
            V_prev = torch.zeros(
                x.shape[0], self.D * self.N,
                device=x.device, dtype=x.dtype,
            )

        I_t = self.W_in(x)

        # β 调制仅依赖 x
        beta = torch.sigmoid(self.W_beta_x(x) + self.b_beta)
        alpha = F.softplus(self.W_alpha_x(x) + self.b_alpha)
        v_th = self.v_th_min + torch.abs(self.W_th_x(x) + self.b_th)

        gate = torch.sigmoid(self.W_gate(x))
        I_skip = self.W_skip(x)

        s_hidden = self.hidden_neuron(I_t, beta, alpha, v_th)

        # 用 V_post（膜电压）做输出投影，与 forward_parallel 一致
        V_post = self.hidden_neuron.v  # 发放+重置后的膜电位
        I_out = self.W_out(V_post)
        I_total = I_out * gate + I_skip

        return I_total  # 连续值


# ============================================================
# Section: snn_decoder_layer
# ============================================================

# ====== Fused halt weight computation (torch.compile) ======
# 7-8 个独立 element-wise kernel → 单 fused kernel
# sigmoid + clamp + log1p + cumsum + exp + normalize
# 首次调用触发 JIT 编译（~秒级），后续调用走缓存

def _fused_geometric_halt_eager(halt_logits):
    """融合计算 PonderNet 几何分布停止权重。

    输入: halt_logits (seq_len, K, batch) — halt_proj 的原始输出
    输出: halt_weights (seq_len, K, batch) — 归一化几何分布权重，sum=1

    数学: p_k = σ(logit_k), S_k = ∏_{j<k}(1-p_j), λ_k = p_k·S_k, λ̂_k = λ_k/Σλ
    """
    p_halt = torch.sigmoid(halt_logits).clamp(min=1e-7, max=1.0 - 1e-7)
    log_1_minus_p = torch.log1p(-p_halt)               # (seq_len, K, batch)
    # Exclusive cumsum: log_survive[:, k, :] = Σ_{j<k} log(1-p_j)
    # 避免 torch.cat: 用 cumsum([:, :-1]) 填充 [:, 1:]
    log_survive = torch.zeros_like(log_1_minus_p)
    log_survive[:, 1:, :] = torch.cumsum(log_1_minus_p[:, :-1, :], dim=1)
    survive = torch.exp(log_survive)                    # (seq_len, K, batch)
    halt_weights = p_halt * survive                     # λ_k = p_k · S_k
    halt_weights = halt_weights / (halt_weights.sum(dim=1, keepdim=True) + 1e-8)
    return halt_weights


_fused_geometric_halt = torch.compile(_fused_geometric_halt_eager, backend='inductor', fullgraph=True)


# ============================================================
# Section: v3 PonderNet (input-conditioned per-token k_t)
# ============================================================

class KPredictor(nn.Module):
    """Input-conditioned one-shot compute-budget predictor.

    Mamba-Δ analog: takes per-token input embedding, outputs K-way logits over
    the "how many PLIF integration steps this token needs" decision.

    Training: Gumbel-Softmax Straight-Through hard-pick + DeepSpeek-V3 style
    gradient-free bias balancing + forced exploration to prevent halt-distribution
    collapse at deep k positions.

    Inference: pure argmax over logits (no Gumbel, no exploration).

    Spec: docs/v3_ponder_spec.md §2.1
    """

    _BIAS_CLAMP = 5.0      # prevent runaway bias drift
    _MIN_USAGE = 1.0 / 1024  # floor to avoid log(0) in stats

    def __init__(self, D: int, K: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(D // 4, 64)
        self.D = D
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(D, h),
            nn.SiLU(),
            nn.Linear(h, K),
        )
        # Gradient-free balancing bias (not a learnable parameter)
        self.register_buffer("bias", torch.zeros(K))
        # Running usage stats for EMA-smoothed bias updates
        self.register_buffer("_usage_ema", torch.full((K,), 1.0 / K))

        # Xavier-init the two linears; small scale so initial logits are near zero
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.weight.data.mul_(0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_emb: (..., D) per-token embedding (un-repeated by K)
        Returns:
            logits: (..., K) halt-step distribution logits (bias added)
        """
        return self.net(input_emb) + self.bias

    @torch.no_grad()
    def update_bias(
        self,
        y_hard: torch.Tensor,
        lr_bias: float,
        ema_decay: float = 0.99,
    ) -> None:
        """Gradient-free bias balancing (DeepSeek-V3 aux-loss-free style).

        Called once per training step AFTER optimizer.step().

        DeepSpeed compat:
          - Buffers are replicated across ranks (not sharded by ZeRO-2).
          - Each rank sees only its own micro-batch in `y_hard`, so without sync
            every rank would drift to a different bias → we all-reduce usage
            across ranks so the update is identical everywhere.

        Args:
            y_hard: (..., K) one-hot selection tensor from this step's forward
            lr_bias: bias learning rate (typical 1e-3)
            ema_decay: EMA smoothing factor for usage statistics
        """
        if y_hard.numel() == 0:
            return
        # Aggregate y_hard over all leading dims except K (cast to fp32 for stable mean)
        usage = y_hard.float().reshape(-1, self.K).mean(dim=0)  # (K,)
        # All-reduce usage across ranks so bias stays in sync
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(usage, op=dist.ReduceOp.AVG)
        # EMA-smooth (fp32 arithmetic via _usage_ema being promoted to fp32)
        self._usage_ema.mul_(ema_decay).add_(usage.to(self._usage_ema.dtype), alpha=1.0 - ema_decay)
        # Deviation from uniform
        diff = self._usage_ema - (1.0 / self.K)
        # Underused → raise bias; overused → lower bias
        self.bias.add_(-lr_bias * diff.to(self.bias.dtype))
        # Clamp to avoid runaway drift
        self.bias.clamp_(-self._BIAS_CLAMP, self._BIAS_CLAMP)


def _ponder_v3_pick(
    k_logits: torch.Tensor,
    training: bool,
    temperature: float,
    eps_explore: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-ST hard-pick (training) or argmax (inference).

    Args:
        k_logits: (..., K) logits (already include KPredictor.bias)
        training: True if in training mode
        temperature: Gumbel temperature (>0)
        eps_explore: forced-exploration probability (0 disables)

    Returns:
        y_st: (..., K) — straight-through tensor (forward=y_hard, backward=y_soft)
        y_hard: (..., K) — the actual one-hot selection (detached)
    """
    K = k_logits.shape[-1]
    if training:
        gumbel = -torch.log(-torch.log(torch.rand_like(k_logits) + 1e-9) + 1e-9)
        logits_noisy = (k_logits + gumbel) / max(temperature, 1e-4)
        y_soft = F.softmax(logits_noisy, dim=-1)
        y_hard = F.one_hot(y_soft.argmax(dim=-1), K).to(y_soft.dtype)

        # Forced exploration: ε probability to replace y_hard with uniform random
        if eps_explore > 0:
            leading_shape = k_logits.shape[:-1]
            random_k = torch.randint(0, K, leading_shape, device=k_logits.device)
            explore_mask = torch.rand(leading_shape, device=k_logits.device) < eps_explore
            y_hard_rand = F.one_hot(random_k, K).to(y_soft.dtype)
            y_hard = torch.where(
                explore_mask.unsqueeze(-1).expand_as(y_hard),
                y_hard_rand,
                y_hard,
            )

        # Straight-through: forward=y_hard, backward gradient flows via y_soft
        y_st = y_hard + y_soft - y_soft.detach()
    else:
        # Inference: pure argmax, no noise
        y_hard = F.one_hot(k_logits.argmax(dim=-1), K).to(k_logits.dtype)
        y_st = y_hard
    return y_st, y_hard.detach()


class SNNDecoderLayer(MemoryModule):
    """
    单个 SNN 解码层（连续残差流 + K 帧聚合版本）。

    层间传递连续值 h (TK, batch, D)，通过 PLIF 神经元提取连续激活（V_post 泄漏量），
    输入 SNN 子层处理后，K 帧聚合为 1 per token，经 out_proj 投影，
    广播回 K 帧做残差连接。

    K 帧聚合使 β 的时间动力学（控制 K 步内的膜电位演化）产生可微分的
    token 级效应，解决 β 梯度为纯噪声的问题。

    Args:
        D: 可见维度
        N: 状态扩展因子
        D_ff: FFN 中间层维度
        v_th_min: SNNBlock 动态阈值下限
        ffn_v_threshold: SNNFFN gate/up 神经元阈值
        K: 每 token 的 SNN 时间步数
        num_layers: 总层数（用于残差输出缩放 + SNNFFN down_proj 缩放）
        layer_idx: 当前层索引
    """

    def __init__(
        self,
        D: int,
        N: int,
        D_ff: int,
        v_th_min: float,
        ffn_v_threshold: float,
        K: int = 16,
        num_layers: int = 1,
        layer_idx: int = 0,
        k_predictor_hidden: int | None = None,
    ):
        super().__init__()
        self.D = D
        self.K = K

        self.snn_block = SNNBlock(
            D=D, N=N, v_th_min=v_th_min,
        )
        self.snn_ffn = SNNFFN(
            D=D, D_ff=D_ff,
            output_v_threshold=ffn_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
        )

        # Pre-LN 分支归一化: h → RMSNorm → PLIFNode
        self.block_norm = RMSNorm(D)
        self.ffn_norm = RMSNorm(D)

        # 输入神经元: RMSNorm(h) → V_post 膜电位激活（D 维可学习 β 和 V_th）
        self.input_neuron1 = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )
        self.input_neuron2 = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # 输出投影（突触）: activation (D) → 输出 (D)
        self.block_out_proj = nn.Linear(D, D, bias=False)
        self.ffn_out_proj = nn.Linear(D, D, bias=False)

        # ====== v3 动态 K: input-conditioned KPredictor（取代 halt_proj） ======
        # 每子层独立一个 KPredictor：block 子层和 FFN 子层看到的 input 不同，
        # 期望它们学不同的 k_t 分配策略。Gumbel-ST 训练 + argmax 推理在
        # _ponder_aggregate_v3 里实现。
        self.block_k_predictor = KPredictor(D=D, K=K, hidden=k_predictor_hidden)
        self.ffn_k_predictor = KPredictor(D=D, K=K, hidden=k_predictor_hidden)

        # 残差输出缩放初始化（GPT-2 style: σ = 0.02 / √(2·num_layers)）
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.block_out_proj.weight, std=std)
        nn.init.normal_(self.ffn_out_proj.weight, std=std)

        # v3 PonderNet runtime state (set by SNNLanguageModel per step)
        self.ponder_T = 1.0       # Gumbel temperature
        self.eps_explore = 0.0    # forced exploration rate

        # Cached y_hard per sub-layer for post-step bias update (detached)
        self._last_y_hard_block = None
        self._last_y_hard_ffn = None

    def _input_neuron_parallel(self, input_neuron, x):
        """
        输入 PLIF 神经元的 parallel scan 前向传播。

        完整 PLIF 动力学: V[t] = β·V[t-1] + (1-β)·x[t], spike = Θ(V-V_th), 软重置。
        输出膜电位泄漏量 (1-β)·V_post 作为激活值——即每步因指数衰减将泄漏的量。
        相比直接传递 V_post，泄漏量自然强调快响应神经元（大 1-β），
        抑制慢记忆神经元（小 1-β），实现隐式的时间尺度加权。

        Args:
            input_neuron: PLIFNode 实例（D 维可学习 β 和 V_th）
            x: (TK, batch, D) — 连续值输入

        Returns:
            leak: (TK, batch, D) — 膜电位泄漏量 (1-β)·V_post
        """
        TK, batch, D = x.shape
        input_dtype = x.dtype

        beta = input_neuron.beta.to(input_dtype)  # fp32→bf16 for compute
        u = (1.0 - beta) * x

        v_init = input_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=x.device, dtype=x.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = input_neuron.v_th.to(input_dtype).unsqueeze(0).expand(batch, D).contiguous()

        output, v_last = plif_rowparam_forward_v2(
            beta_row, u, v_th_row, v_init,
        )

        input_neuron.v = v_last.detach()
        # Diagnostic: 发放率 (bio-ReLU: output > 0 ≡ fired)
        with torch.no_grad():
            input_neuron._last_firing_rate = (output > 0).float().mean().item()
        return output.to(input_dtype)  # 超阈电流 (sparse, bio-ReLU)

    def _ponder_aggregate_v3(
        self,
        frames: torch.Tensor,
        input_emb: torch.Tensor,
        k_predictor: "KPredictor",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """v3 PonderNet: input-conditioned hard-pick of k_t per token.

        Spec: docs/v3_ponder_spec.md §3.2

        Training:
            1. k_predictor(input_emb) → K-way logits (含 bias)
            2. Gumbel-Softmax Straight-Through 硬采样 k_t (forward=one-hot)
            3. output = Σ y_st[k] · frame[k] 前向值 = frame[k_t], 反向梯度经 y_soft
            4. 三层反坍缩：Gumbel 温度 + forced exploration + bias balancing（后者在 update_bias）

        Inference:
            pure argmax 单步选择，无 Gumbel / exploration

        Args:
            frames: (seq_len, K, batch, D) — SNN 子层 K 帧输出
            input_emb: (seq_len, batch, D) — 该子层 token-level 输入 (K 维之前)
            k_predictor: 该子层对应的 KPredictor 模块

        Returns:
            aggregated: (seq_len, batch, D) — 单一 k_t 位置的 frame
            ponder_cost: scalar — E[k_t]（log 用，Stage 1 不进 loss）
            expected_k: (seq_len, batch) — 每 token 的 E[k_t]（诊断用, detached）
            y_hard: (seq_len, batch, K) — 本次选择的 one-hot（detached，给 bias update）
        """
        seq_len, K, batch, D = frames.shape

        # Step 1: per-token input-conditioned logits
        k_logits = k_predictor(input_emb)              # (seq_len, batch, K)

        # Step 2: Gumbel-ST (train) / argmax (eval)
        y_st, y_hard = _ponder_v3_pick(
            k_logits,
            training=self.training,
            temperature=self.ponder_T,
            eps_explore=self.eps_explore,
        )                                                # y_st: (seq_len, batch, K)

        # Step 3: 硬选聚合 output = Σ y_st[k] · frame[k]
        # y_st: (seq_len, batch, K) → (seq_len, K, batch, 1) for broadcast
        y_st_b = y_st.permute(0, 2, 1).unsqueeze(-1)    # (seq_len, K, batch, 1)
        aggregated = (y_st_b * frames).sum(dim=1)       # (seq_len, batch, D)

        # Step 4: 期望 k_t 统计（diagnostic / log 用）
        steps = torch.arange(1, K + 1, device=frames.device, dtype=frames.dtype)
        expected_k = (y_st.detach() * steps[None, None, :]).sum(-1)  # (seq_len, batch)
        ponder_cost = expected_k.mean()

        return aggregated, ponder_cost, expected_k.detach(), y_hard

    def forward_parallel(self, h):
        """
        并行前向传播：连续残差流 + v3 PonderNet 硬选 k_t。

        每子层流程:
          1. Pre-LN + PLIF(V_post) 输入神经元 → SNN 子层 K 帧输出
          2. v3 PonderNet: k_predictor(input_emb) → Gumbel-ST 硬选 k_t 位置
          3. output = frame[k_t] (单一时刻膜电位状态)
          4. out_proj + 残差中心化 → 广播回 TK

        Args:
            h: (TK, batch, D) — 连续值输入

        Returns:
            h: (TK, batch, D) — 连续值输出
            ponder_cost: scalar — 两个子层的平均 E[k_t]（仅 log 用）
        """
        TK, batch, D = h.shape
        K = self.K
        seq_len = TK // K

        # ==== 子层 1: SNNBlock ====
        # input_emb 取第 0 帧作为 token-level 输入给 k_predictor
        h_block_input = h.view(seq_len, K, batch, D)[:, 0]  # (seq_len, batch, D)
        v_in = self._input_neuron_parallel(self.input_neuron1, self.block_norm(h))
        cont_block = self.snn_block.forward_parallel(v_in)  # (TK, batch, D)

        frames_block = cont_block.view(seq_len, K, batch, D)
        combined_block, pc_block, ek_block, y_hard_block = self._ponder_aggregate_v3(
            frames_block, h_block_input, self.block_k_predictor,
        )
        res_block = self.block_out_proj(combined_block)  # (seq_len, batch, D)
        res_block = res_block - res_block.mean(dim=-1, keepdim=True)
        h = h + res_block.repeat_interleave(K, dim=0)

        # ==== 子层 2: SNNFFN ====
        h_ffn_input = h.view(seq_len, K, batch, D)[:, 0]
        v_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h))
        cont_ffn = self.snn_ffn.forward_parallel(v_in2)  # (TK, batch, D)

        frames_ffn = cont_ffn.view(seq_len, K, batch, D)
        combined_ffn, pc_ffn, ek_ffn, y_hard_ffn = self._ponder_aggregate_v3(
            frames_ffn, h_ffn_input, self.ffn_k_predictor,
        )
        res_ffn = self.ffn_out_proj(combined_ffn)
        res_ffn = res_ffn - res_ffn.mean(dim=-1, keepdim=True)
        h = h + res_ffn.repeat_interleave(K, dim=0)

        ponder_cost = (pc_block + pc_ffn) / 2.0

        # Cache y_hard for post-step bias update
        self._last_y_hard_block = y_hard_block
        self._last_y_hard_ffn = y_hard_ffn

        # Diagnostic: per-token E[k_t] range
        with torch.no_grad():
            all_ek = torch.cat([ek_block.flatten(), ek_ffn.flatten()])
            self._ek_min = all_ek.min().item()
            self._ek_max = all_ek.max().item()

        return h, ponder_cost

    def update_bias(self, lr_bias: float = 1e-3, ema_decay: float = 0.99) -> None:
        """Apply gradient-free bias balancing to both sub-layer k_predictors.

        Call once per training step AFTER optimizer.step().
        """
        if self._last_y_hard_block is not None:
            self.block_k_predictor.update_bias(
                self._last_y_hard_block, lr_bias=lr_bias, ema_decay=ema_decay,
            )
        if self._last_y_hard_ffn is not None:
            self.ffn_k_predictor.update_bias(
                self._last_y_hard_ffn, lr_bias=lr_bias, ema_decay=ema_decay,
            )

    def single_step_forward(self, h):
        """
        单步前向传播：连续残差流。

        注意：单步模式无法做动态 K 聚合（每步独立处理）。
        训练和推理均使用 forward_parallel（含动态 K 聚合）。
        此方法仅用于调试。

        Args:
            h: (batch, D) — 连续值输入

        Returns:
            h: (batch, D) — 连续值输出
            ponder_cost: scalar — 0.0（单步无 ponder cost）
        """
        # 子层 1: SNNBlock — RMSNorm → PLIFNode → SNNBlock → out_proj → 残差
        _ = self.input_neuron1(self.block_norm(h))  # 触发 PLIF 动力学，更新 .v
        v_in = self.input_neuron1.v
        v_in = (1.0 - self.input_neuron1.beta) * v_in
        cont_block = self.snn_block.single_step_forward(v_in)
        res_block = self.block_out_proj(cont_block)
        h = h + res_block - res_block.mean(dim=-1, keepdim=True)

        # 子层 2: SNNFFN — RMSNorm → PLIFNode → SNNFFN → out_proj → 残差
        _ = self.input_neuron2(self.ffn_norm(h))
        v_in2 = self.input_neuron2.v
        v_in2 = (1.0 - self.input_neuron2.beta) * v_in2
        cont_ffn = self.snn_ffn.single_step_forward(v_in2)
        res_ffn = self.ffn_out_proj(cont_ffn)
        h = h + res_ffn - res_ffn.mean(dim=-1, keepdim=True)

        return h, torch.tensor(0.0, device=h.device)


# ============================================================
# Section: snn_attention_decoder_layer
# ============================================================

def _precompute_rope_freqs(dim, max_seq_len=8192, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()

def _apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class SNNAttentionDecoderLayer(MemoryModule):
    """
    完整的 SNN-Attention 解码层, 结构对齐 SNNDecoderLayer。

    子层 1: SNN-Attention (token 级 cumsum + RoPE + PLIFNode gate)
    子层 2: SNNFFN (和 SNNDecoderLayer 完全相同)

    Args:
        D: 可见维度
        N: FFN 的状态扩展因子 (子层 2 用)
        D_ff: FFN 中间层维度
        D_key: SNN-Attention 的键/查询维度
        D_value: SNN-Attention 的值维度
        v_th_min: 动态阈值下限
        ffn_v_threshold: SNNFFN 神经元阈值
        K: 每 token SNN 帧数
        num_layers: 总层数 (用于输出缩放)
        layer_idx: 当前层索引
    """

    def __init__(
        self,
        D: int,
        N: int,
        D_ff: int,
        D_key: int = 64,
        D_value: int = 64,
        v_th_min: float = 0.1,
        ffn_v_threshold: float = 0.15,
        K: int = 12,
        num_layers: int = 1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.D = D
        self.K = K
        self.D_key = D_key
        self.D_value = D_value

        # ====== 子层 1: SNN-Attention ======
        self.attn_norm = RMSNorm(D)
        self.qkv_proj = nn.Linear(D, D_key * 2 + D_value, bias=False)
        self.gate_neuron = PLIFNode(
            dim=D, init_tau=2.0, v_threshold=0.8,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )
        self.attn_out_norm = RMSNorm(D_value)
        self.attn_out_proj = nn.Linear(D_value, D, bias=False)

        # RoPE
        assert D_key % 2 == 0
        rope_cos, rope_sin = _precompute_rope_freqs(D_key)
        self.register_buffer('rope_cos', rope_cos, persistent=False)
        self.register_buffer('rope_sin', rope_sin, persistent=False)

        # 持久状态
        self.register_memory('M_state', 0.)

        # ====== 子层 2: SNNFFN (完全复用 SNNDecoderLayer 的结构) ======
        self.ffn_norm = RMSNorm(D)
        self.input_neuron2 = PLIFNode(
            dim=D, init_tau=2.0, v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )
        self.snn_ffn = SNNFFN(
            D=D, D_ff=D_ff,
            output_v_threshold=ffn_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
        )
        self.ffn_out_proj = nn.Linear(D, D, bias=False)

        # v3 PonderNet: FFN 子层的 KPredictor (海马体层只有 FFN 子层需要 K 聚合,
        # attention 子层是 T 维 cumsum 不涉及 K 聚合)
        self.ffn_k_predictor = KPredictor(D=D, K=K, hidden=None)

        # ====== 初始化 ======
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.normal_(self.attn_out_proj.weight, std=std)
        nn.init.normal_(self.ffn_out_proj.weight, std=std)

        # v3 runtime state
        self.ponder_T = 1.0
        self.eps_explore = 0.0
        self._last_y_hard_ffn = None

    def _input_neuron_parallel(self, input_neuron, x):
        """PLIFNode parallel scan, 返回激活值。复用 SNNDecoderLayer 的逻辑。"""
        TK, batch, D = x.shape
        input_dtype = x.dtype
        beta = input_neuron.beta.to(input_dtype)
        u = (1.0 - beta) * x

        v_init = input_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=x.device, dtype=input_dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = input_neuron.v_th.to(input_dtype).unsqueeze(0).expand(batch, D).contiguous()

        output, v_last = plif_rowparam_forward_v2(
            beta_row, u, v_th_row, v_init,
        )
        input_neuron.v = v_last.detach()
        with torch.no_grad():
            input_neuron._last_firing_rate = (output > 0).float().mean().item()
        return output.to(input_dtype)

    def _gate_neuron_parallel(self, h_normed):
        """PLIFNode gate 的 parallel scan, 返回标量门控 (bio-ReLU 超阈电流)。"""
        seq_len, batch, D = h_normed.shape
        input_dtype = h_normed.dtype
        beta_g = self.gate_neuron.beta.to(input_dtype)
        u_g = (1.0 - beta_g) * h_normed

        v_init_g = self.gate_neuron.v
        if isinstance(v_init_g, float):
            v_init_g = torch.zeros(batch, D, device=h_normed.device, dtype=input_dtype)

        beta_row = beta_g.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = self.gate_neuron.v_th.to(input_dtype).unsqueeze(0).expand(batch, D).contiguous()

        gate_out, v_last_g = plif_rowparam_forward_v2(
            beta_row, u_g, v_th_row, v_init_g,
        )
        with torch.no_grad():
            self.gate_neuron._last_firing_rate = (gate_out > 0).float().mean().item()
        self.gate_neuron.v = v_last_g.detach()
        # gate_out 已经是超阈电流 (sparse, ReLU), 直接做 mean pool 作为标量门控
        return gate_out.mean(dim=-1, keepdim=True).to(input_dtype)

    def _ponder_aggregate_v3(
        self,
        frames: torch.Tensor,
        input_emb: torch.Tensor,
        k_predictor: "KPredictor",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """复用 SNNDecoderLayer 的 v3 PonderNet 聚合 (硬选 k_t)。"""
        seq_len, K, batch, D = frames.shape
        k_logits = k_predictor(input_emb)
        y_st, y_hard = _ponder_v3_pick(
            k_logits,
            training=self.training,
            temperature=self.ponder_T,
            eps_explore=self.eps_explore,
        )
        y_st_b = y_st.permute(0, 2, 1).unsqueeze(-1)
        aggregated = (y_st_b * frames).sum(dim=1)
        steps = torch.arange(1, K + 1, device=frames.device, dtype=frames.dtype)
        expected_k = (y_st.detach() * steps[None, None, :]).sum(-1)
        ponder_cost = expected_k.mean()
        return aggregated, ponder_cost, expected_k.detach(), y_hard

    def forward_parallel(self, h):
        """
        并行前向传播, 返回 (h, ponder_cost) 对齐 SNNDecoderLayer 接口。
        """
        TK, batch, D = h.shape
        K = self.K
        seq_len = TK // K

        # ====== 子层 1: SNN-Attention ======
        # K 帧平均聚合 → token 级
        h_token = h.view(seq_len, K, batch, D).mean(dim=1)
        h_normed = self.attn_norm(h_token)

        # 投影 q, k, v
        flat = h_normed.reshape(seq_len * batch, D)
        qkv = self.qkv_proj(flat)
        q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
        q = q.reshape(seq_len, batch, self.D_key)
        k = k.reshape(seq_len, batch, self.D_key)
        v = v.reshape(seq_len, batch, self.D_value)

        # RoPE（转成输入 dtype，兼容 bf16 推理）
        rope_cos = self.rope_cos[:seq_len].unsqueeze(1).to(q.dtype)
        rope_sin = self.rope_sin[:seq_len].unsqueeze(1).to(q.dtype)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        # PLIFNode gate
        gate = self._gate_neuron_parallel(h_normed)

        # L2 归一化 k
        k = F.normalize(k, dim=-1)

        # 无衰减累积 M
        kv_outer = k.unsqueeze(-1) * v.unsqueeze(-2)
        kv_gated = gate.unsqueeze(-1) * kv_outer
        M_all = torch.cumsum(kv_gated, dim=0)

        if not isinstance(self.M_state, float):
            M_all = M_all + self.M_state.unsqueeze(0)
        self.M_state = M_all[-1].detach()

        # 读出 + 归一化 + 投影
        attn_out = torch.einsum('sbk,sbkv->sbv', q, M_all)
        attn_out = self.attn_out_norm(attn_out)
        res_attn = self.attn_out_proj(attn_out.reshape(seq_len * batch, self.D_value))
        res_attn = res_attn.reshape(seq_len, batch, D)
        res_attn = res_attn - res_attn.mean(dim=-1, keepdim=True)  # 残差中心化

        # 广播回 TK 帧 + 残差
        h = h + res_attn.unsqueeze(1).expand(-1, K, -1, -1).reshape(TK, batch, D)

        # ====== 子层 2: SNNFFN (v3 PonderNet) ======
        h_ffn_input = h.view(seq_len, K, batch, D)[:, 0]
        v_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h))
        cont_ffn = self.snn_ffn.forward_parallel(v_in2)

        frames_ffn = cont_ffn.view(seq_len, K, batch, D)
        combined_ffn, pc_ffn, ek_ffn, y_hard_ffn = self._ponder_aggregate_v3(
            frames_ffn, h_ffn_input, self.ffn_k_predictor,
        )
        res_ffn = self.ffn_out_proj(combined_ffn)
        res_ffn = res_ffn - res_ffn.mean(dim=-1, keepdim=True)

        h = h + res_ffn.repeat_interleave(K, dim=0)

        # Cache y_hard for bias update
        self._last_y_hard_ffn = y_hard_ffn

        # 存储 E[K] 诊断
        with torch.no_grad():
            self._ek_min = ek_ffn.min().item()
            self._ek_max = ek_ffn.max().item()

        return h, pc_ffn

    def update_bias(self, lr_bias: float = 1e-3, ema_decay: float = 0.99) -> None:
        """Gradient-free bias balancing for the FFN sub-layer k_predictor."""
        if self._last_y_hard_ffn is not None:
            self.ffn_k_predictor.update_bias(
                self._last_y_hard_ffn, lr_bias=lr_bias, ema_decay=ema_decay,
            )


# ============================================================
# Section: model
# ============================================================

@dataclass
class SNNModelOutput:
    """模型输出容器，对齐教程 CausalLMOutputWithPast 接口。"""
    last_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ponder_cost: Optional[torch.Tensor] = None  # 动态 K: 平均期望步数

class SNNLanguageModel(nn.Module):
    """
    从零训练的 SNN 隐状态空间语言模型（parallel scan）。

    Args:
        vocab_size: 词表大小（默认 64000）
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 最大 SNN 时间步数（K_max）。PonderNet 动态决定有效步数 ∈ [1, K]。
           K 越大 → 复杂 token 可用更多步数，但计算量和显存线性增长。
        num_layers: SNN 解码层数
        D_ff: FFN 中间层维度
        v_th_min: 动态阈值下限
    """

    def __init__(
        self,
        vocab_size: int = 64000,
        D: int = 1024,
        N: int = 8,
        K: int = 12,
        num_layers: int = 24,
        D_ff: int = 3072,
        v_th_min: float = 0.1,
        memory_layer_interval: int = 4,  # 0=禁用联想记忆层
        D_key: int = 128,
        D_value: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff
        self.memory_layer_interval = memory_layer_interval
        self.v_th_min = v_th_min
        self.D_key = D_key
        self.D_value = D_value

        # ====== Embedding + Norm（全部可训练）======
        self.embed_tokens = nn.Embedding(vocab_size, D)
        self.norm = LateralInhibition(D)

        # ====== 解码投影 ======
        self.decode_proj = nn.Linear(D, D, bias=False)

        # ====== 输出 RMSNorm + 输出神经元 ======
        self.output_norm = RMSNorm(D)
        self.output_neuron = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.3,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # ====== 混合层栈: SNN Decoder + SNN-Attention Decoder ======
        # 每 memory_layer_interval 层插入 1 层 SNN-Attention 解码层
        # 例: interval=4, 24 层 → 层 3,7,11,15,19,23 为 SNN-Attention 层
        self.layers = nn.ModuleList()
        self.layer_types = []  # 'snn' or 'memory'
        for i in range(num_layers):
            if memory_layer_interval > 0 and (i + 1) % memory_layer_interval == 0:
                self.layers.append(SNNAttentionDecoderLayer(
                    D=D, N=N, D_ff=D_ff,
                    D_key=D_key, D_value=D_value,
                    v_th_min=v_th_min,
                    ffn_v_threshold=0.15,
                    K=K,
                    num_layers=num_layers,
                    layer_idx=i,
                ))
                self.layer_types.append('memory')
            else:
                self.layers.append(SNNDecoderLayer(
                    D=D, N=N, D_ff=D_ff, v_th_min=v_th_min,
                    ffn_v_threshold=0.15,
                    K=K,
                    num_layers=num_layers,
                    layer_idx=i,
                ))
                self.layer_types.append('snn')

        self._init_weights()

    def _init_weights(self):
        """初始化所有可训练权重（从零训练）。"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.decode_proj.weight)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入边界：token_ids → 连续值序列。

        Embedding lookup，每 token 重复 K 次作为 SNN 时间步输入。
        梯度可通过 embedding 直接反传。

        Returns: (seq_len*K, batch, D), 连续值
        """
        emb = self.embed_tokens(token_ids)       # (batch, seq_len, D)
        batch, seq_len, D = emb.shape
        # 每 token 重复 K 次: (batch, seq_len, D) → (batch, seq_len*K, D) → (TK, batch, D)
        emb_k = emb.unsqueeze(2).expand(-1, -1, self.K, -1).reshape(batch, seq_len * self.K, D)
        return emb_k.permute(1, 0, 2).contiguous()  # (TK, batch, D)

    def set_ponder_temperature(self, T: float) -> None:
        """Set Gumbel temperature on every layer's KPredictor path.

        Call from training loop each step to anneal T_init → T_final.
        """
        for layer in self.layers:
            if hasattr(layer, "ponder_T"):
                layer.ponder_T = float(T)

    def set_ponder_exploration(self, eps: float) -> None:
        """Set forced-exploration probability on every layer."""
        for layer in self.layers:
            if hasattr(layer, "eps_explore"):
                layer.eps_explore = float(eps)

    def update_ponder_bias(self, lr_bias: float = 1e-3, ema_decay: float = 0.99) -> None:
        """Apply DeepSeek-V3 style gradient-free bias balancing across all layers.

        Call once per training step AFTER optimizer.step().
        """
        for layer in self.layers:
            if hasattr(layer, "update_bias"):
                layer.update_bias(lr_bias=lr_bias, ema_decay=ema_decay)

    def snn_forward(self, h_seq: torch.Tensor):
        """SNN 核心：h_seq → (h_out, ponder_cost)。

        纯 SNN 层计算，带梯度检查点。
        每层返回 (h, ponder_cost)，ponder_cost 作为 checkpoint 输出保留梯度图。

        Returns:
            h: (seq_len*K, batch, D), 连续值
            total_ponder_cost: scalar, 所有层平均期望步数
        """
        h = h_seq
        ponder_costs = []

        def _layer_forward(layer_mod, x):
            functional.reset_net(layer_mod)
            return layer_mod.forward_parallel(x)  # 统一返回 (h, ponder_cost)

        for layer_module in self.layers:
            # use_reentrant=False 在 PyTorch 2.x 默认自动 preserve RNG state,
            # 所以 Gumbel-Softmax 的 torch.rand 在 forward / backward recompute
            # 产生相同随机序列 → _last_y_hard_block/ffn 在 recompute 后仍是原值.
            h, pc = checkpoint(
                _layer_forward, layer_module, h,
                use_reentrant=False,
            )
            ponder_costs.append(pc)

        total_ponder_cost = sum(ponder_costs) / len(ponder_costs)
        return h, total_ponder_cost

    def _output_neuron_parallel(self, h: torch.Tensor) -> torch.Tensor:
        """输出 PLIF 神经元的 parallel scan 前向：连续 h → 膜电位泄漏量。

        Args:
            h: (TK, batch, D) 连续值（SNN 最后一层输出）

        Returns:
            leak: (TK, batch, D) 膜电位泄漏量 (1-β)·V_post
        """
        TK, batch, D = h.shape
        input_dtype = h.dtype

        beta = self.output_neuron.beta.to(input_dtype)
        u = (1.0 - beta) * h

        v_init = self.output_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=h.device, dtype=input_dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = self.output_neuron.v_th.to(input_dtype).unsqueeze(0).expand(batch, D).contiguous()

        output, v_last = plif_rowparam_forward_v2(
            beta_row, u, v_th_row, v_init,
        )

        self.output_neuron.v = v_last.detach()
        with torch.no_grad():
            self.output_neuron._last_firing_rate = (output > 0).float().mean().item()
        # 输出神经元: 传超阈电流 (sparse, bio-ReLU)
        return output.to(input_dtype)

    def decode(self, h_out: torch.Tensor, seq_len: int) -> torch.Tensor:
        """输出边界：连续 h → 输出神经元(V_post) → K 帧聚合 → logits。

        梯度流: loss → logits → norm → decode_proj → K帧mean
                → V_post(output_neuron) → h_out → SNN layers

        Returns: (batch, seq_len, vocab_size)
        """
        h_out = self.output_norm(h_out)                    # RMSNorm: 控制 scale
        v_out = self._output_neuron_parallel(h_out)    # (TK, batch, D), V_post 膜电位
        # K 帧聚合: (TK, batch, D) → (seq_len, K, batch, D) → mean → (seq_len, batch, D)
        decoded = v_out.view(seq_len, self.K, -1, self.D).mean(dim=1)
        decoded = decoded.permute(1, 0, 2)                 # (batch, seq_len, D)
        h = self.decode_proj(decoded)                      # (batch, seq_len, D)
        h = self.norm(h)                                   # (batch, seq_len, D)
        return F.linear(h, self.embed_tokens.weight)       # (batch, seq_len, vocab)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成（SNN 神经元状态跨 token 连续维护）。

        1. Prefill: forward_parallel 并行处理 prompt，建立所有神经元 V 状态
        2. Autoregressive: 逐 token 生成，每 token 用 forward_parallel 处理 K 帧
           复用 Triton parallel scan kernel，神经元 V 状态跨 token 连续传递

        Args:
            prompt_ids: (batch, prompt_len) token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（<=0 = greedy）
            top_k: top-k 采样（None/0 = 不限制）
            top_p: nucleus 采样阈值（1.0 = 不限制）
            repetition_penalty: 重复惩罚（1.0 = 无惩罚，>1.0 = 惩罚重复）
            eos_token_id: 遇到此 token 停止生成

        Returns:
            (batch, prompt_len + generated_len) 完整序列
        """
        batch, prompt_len = prompt_ids.shape

        # 重置所有神经元（新序列的初始条件 V=0）
        for layer_module in self.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.output_neuron)

        # ====== Prefill: parallel 处理整个 prompt ======
        h_seq = self.encode(prompt_ids)  # (prompt_len*K, batch, D), 连续值
        h = h_seq
        for layer_module in self.layers:
            h, _ = layer_module.forward_parallel(h)
        # 此时所有层的所有神经元 .v 状态 = prompt 末尾状态

        logits = self.decode(h, prompt_len)

        # 已生成的 token ID 集合（用于 repetition_penalty）
        generated_ids = prompt_ids.clone()

        # 采样第一个新 token
        next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p,
                                  repetition_penalty, generated_ids)
        generated = [next_token]
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # ====== Autoregressive: 逐 token，forward_parallel 处理 K 帧 ======
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 编码单 token → K 帧连续值（复用 encode）
            frames = self.encode(next_token)  # (K, batch, D)

            # K 帧通过 SNN — 不 reset，神经元 .v 跨 token 连续传递
            h = frames
            for layer_module in self.layers:
                h, _ = layer_module.forward_parallel(h)

            logits = self.decode(h, 1)

            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p,
                                      repetition_penalty, generated_ids)
            generated.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0,
                top_k: int = None, top_p: float = 1.0,
                repetition_penalty: float = 1.0,
                generated_ids: torch.Tensor = None) -> torch.Tensor:
        """从 logits 采样（temperature + repetition_penalty + top-k + top-p）。

        Returns: (batch, 1)
        """
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        # Repetition penalty: 对已出现的 token 降低概率
        if repetition_penalty != 1.0 and generated_ids is not None:
            for b in range(logits.size(0)):
                prev_ids = generated_ids[b].unique()
                score = logits[b, prev_ids]
                # 正 logit 除以 penalty（降低），负 logit 乘以 penalty（更负）
                logits[b, prev_ids] = torch.where(
                    score > 0, score / repetition_penalty, score * repetition_penalty
                )

        logits = logits / temperature

        # Top-k
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 移除累积概率超过 top_p 的 token（保留第一个超过的）
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        前向传播（全膜电位 + 动态 K）。

        encode → h_seq               # 输入（embed repeat K 次，可微分）
        snn_forward → h_out, pc      # SNN 核心（全膜电位 + 动态 K 聚合）
        decode → logits              # 输出（V_post → K帧mean → proj → logits）

        梯度流:
          embed_tokens → repeat K → SNN layers(V_post + 动态K)
            → output_neuron(V_post) → K帧mean → decode_proj → logits(tied head)
          ponder_cost: 动态 K 正则化，鼓励用更少步数处理简单 token
        """
        batch, seq_len = token_ids.shape

        # 重置所有神经元状态
        for layer_module in self.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.output_neuron)

        # 三段式
        h_seq = self.encode(token_ids)                # 输入边界
        h_out, ponder_cost = self.snn_forward(h_seq)  # SNN 核心 + ponder cost
        logits = self.decode(h_out, seq_len)          # 输出边界

        if target_ids is not None:
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = target_ids.reshape(-1)
            self.last_loss = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=0, reduction='none',
            )
            return SNNModelOutput(
                last_loss=self.last_loss,
                ponder_cost=ponder_cost,
            )

        return SNNModelOutput(logits=logits, ponder_cost=ponder_cost)

    def compensate_modulation_gradients(self, max_comp: float = 100.0):
        """
        Natural Gradient 补偿（两阶段）。

        Phase 1: Sigmoid/softplus 饱和补偿
          β = sigmoid(b_beta), sigmoid 在高 β 区（β=0.99, sigmoid'=0.01）梯度衰减 100x。
          补偿: grad /= activation'(b)，等价于在 β/α 空间做梯度下降。

        Phase 2: 层间梯度均衡
          残差链反向传播每层放大 ~1.17×，num_layers 层累积显著梯度比。
          深层选择性参数（b_beta/b_alpha/b_th）梯度被压制，无法有效学习。
          修复: 将每层调制参数梯度 norm 归一化到所有层的几何均值。

        调用时机: scaler.unscale_(optimizer) 之后、clip_grad_norm_ 之前。

        Args:
            max_comp: 补偿因子上限（防止极端值导致不稳定）
        """
        # ====== Phase 1: Sigmoid/softplus 饱和补偿 ======
        for layer_module in self.layers:
            if not hasattr(layer_module, 'snn_block'):
                continue  # 联想记忆层无调制参数
            block = layer_module.snn_block

            # b_beta: sigmoid 饱和补偿
            # sigmoid'(z) = sigmoid(z) · (1 - sigmoid(z)) = β · (1-β)
            if block.b_beta.grad is not None:
                with torch.no_grad():
                    beta = torch.sigmoid(block.b_beta.data)
                    sigmoid_deriv = (beta * (1.0 - beta)).clamp(min=1.0 / max_comp)
                    block.b_beta.grad.div_(sigmoid_deriv)

            # b_alpha: softplus 补偿（较温和，softplus'(z) = sigmoid(z)）
            if block.b_alpha.grad is not None:
                with torch.no_grad():
                    softplus_deriv = torch.sigmoid(block.b_alpha.data).clamp(min=0.1)
                    block.b_alpha.grad.div_(softplus_deriv)

            # b_th: |·| 导数为 ±1，无衰减，不需要补偿

        # ====== Phase 2: 层间梯度均衡 ======
        # 残差链 h = h + sublayer(h) 的反向路径 ∂h_{l+1}/∂h_l = I + ∂sublayer/∂h_l
        # 每层放大 ~1.17×, 深层累积显著梯度比 → L0 梯度远大于 L_{num_layers-1}
        # 用几何均值归一化每层调制参数梯度 norm，消除残差放大效应
        with torch.no_grad():
            for param_name in ['b_beta', 'b_alpha', 'b_th']:
                norms = []
                params_list = []
                for layer_module in self.layers:
                    if not hasattr(layer_module, 'snn_block'):
                        continue
                    p = getattr(layer_module.snn_block, param_name)
                    if p.grad is not None:
                        n = p.grad.norm().item()
                        if n > 1e-12:
                            norms.append(n)
                            params_list.append(p)

                if len(norms) >= 2:
                    # 几何均值: exp(mean(log(norms))) — 对数尺度均衡，不受极端值影响
                    log_mean = sum(math.log(n) for n in norms) / len(norms)
                    geo_mean = math.exp(log_mean)
                    for p, n in zip(params_list, norms):
                        scale = geo_mean / n
                        scale = max(min(scale, max_comp), 1.0 / max_comp)
                        p.grad.mul_(scale)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        按功能分组的可训练参数。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.gain],
            'decode': list(self.decode_proj.parameters()),
            # 输出神经元
            'output_neuron': [self.output_neuron.w, self.output_neuron.v_th],
            # RMSNorm（Pre-LN 分支归一化）
            'rms_norms': [self.output_norm.weight],
            # 残差流组件
            'residual_projs': [],
            'input_neurons': [],
            # v3 动态 K: input-conditioned KPredictor 参数 (取代 halt_projs)
            'k_predictors': [],
            # SNNBlock 参数
            'W_in': [],
            'W_beta': [],
            'W_alpha': [],
            'W_th': [],
            'W_gate': [],
            'W_skip': [],
            'W_out': [],
            'conv1d': [],               # Mamba-style depthwise short conv (3D weight)
            'b_beta': [],
            'b_alpha': [],
            'b_th': [],
            'block_output_neuron': [],
            # SNNFFN 参数
            'ffn_gate_proj': [],
            'ffn_up_proj': [],
            'ffn_down_proj': [],
            'ffn_skip_proj': [],
            'ffn_neurons': [],
        }

        for layer_module, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'memory':
                # SNNAttentionDecoderLayer: attn 子层 + ffn 子层
                groups['residual_projs'].extend([
                    layer_module.qkv_proj.weight,
                    layer_module.attn_out_proj.weight,
                    layer_module.ffn_out_proj.weight,
                ])
                groups['input_neurons'].extend([
                    layer_module.gate_neuron.w, layer_module.gate_neuron.v_th,
                    layer_module.input_neuron2.w, layer_module.input_neuron2.v_th,
                ])
                groups['rms_norms'].extend([
                    layer_module.attn_norm.weight,
                    layer_module.attn_out_norm.weight,
                    layer_module.ffn_norm.weight,
                ])
                groups['k_predictors'].extend(list(layer_module.ffn_k_predictor.net.parameters()))
                # SNNFFN 参数
                ffn = layer_module.snn_ffn
                groups['ffn_gate_proj'].append(ffn.gate_proj.weight)
                groups['ffn_up_proj'].append(ffn.up_proj.weight)
                groups['ffn_down_proj'].append(ffn.down_proj.weight)
                groups['ffn_skip_proj'].append(ffn.skip_proj.weight)
                groups['ffn_neurons'].extend([
                    ffn.gate_neuron.w, ffn.gate_neuron.v_th,
                    ffn.up_neuron.w, ffn.up_neuron.v_th,
                ])
                continue

            block = layer_module.snn_block
            ffn = layer_module.snn_ffn

            # 残差流组件
            groups['residual_projs'].extend([
                layer_module.block_out_proj.weight,
                layer_module.ffn_out_proj.weight,
            ])
            groups['input_neurons'].extend([
                layer_module.input_neuron1.w,
                layer_module.input_neuron1.v_th,
                layer_module.input_neuron2.w,
                layer_module.input_neuron2.v_th,
            ])
            groups['rms_norms'].extend([
                layer_module.block_norm.weight,
                layer_module.ffn_norm.weight,
            ])

            # v3 KPredictor 参数（仅 net 可训练，bias buffer 不在 autograd 图）
            groups['k_predictors'].extend(list(layer_module.block_k_predictor.net.parameters()))
            groups['k_predictors'].extend(list(layer_module.ffn_k_predictor.net.parameters()))

            # SNNBlock 参数
            groups['W_in'].append(block.W_in.weight)
            groups['W_beta'].extend([block.W_beta_x.weight])
            groups['W_alpha'].extend([block.W_alpha_x.weight])
            groups['W_th'].extend([block.W_th_x.weight])
            groups['W_gate'].append(block.W_gate.weight)
            groups['W_skip'].append(block.W_skip.weight)
            groups['W_out'].append(block.W_out.weight)
            # Depthwise short conv (Mamba-style). 3D weight (C, 1, kernel) — not
            # matrix-orthogonalizable, so exclude from Muon group in downstream.
            groups['conv1d'].append(block.conv1d.weight)
            if block.conv1d.bias is not None:
                groups['conv1d'].append(block.conv1d.bias)
            groups['b_beta'].append(block.b_beta)
            groups['b_alpha'].append(block.b_alpha)
            groups['b_th'].append(block.b_th)

            # SNNFFN 参数
            groups['ffn_gate_proj'].append(ffn.gate_proj.weight)
            groups['ffn_up_proj'].append(ffn.up_proj.weight)
            groups['ffn_down_proj'].append(ffn.down_proj.weight)
            groups['ffn_skip_proj'].append(ffn.skip_proj.weight)
            groups['ffn_neurons'].extend([
                ffn.gate_neuron.w, ffn.gate_neuron.v_th,
                ffn.up_neuron.w, ffn.up_neuron.v_th,
            ])

        return groups


# ============================================================
# Section: HF Wrapper
# ============================================================

class NeuronSparkForCausalLM(PreTrainedModel):
    config_class = NeuronSparkConfig
    supports_gradient_checkpointing = False
    _tied_weights_keys = []
    all_tied_weights_keys = {}

    def __init__(self, config: NeuronSparkConfig):
        super().__init__(config)
        self.snn = SNNLanguageModel(
            vocab_size=config.vocab_size,
            D=config.D,
            N=config.N,
            K=config.K,
            num_layers=config.num_layers,
            D_ff=config.D_ff,
            v_th_min=config.v_th_min,
            memory_layer_interval=config.memory_layer_interval,
            D_key=config.D_key,
            D_value=config.D_value,
        )

    def get_input_embeddings(self):
        return self.snn.embed_tokens

    def set_input_embeddings(self, value):
        self.snn.embed_tokens = value

    def get_output_embeddings(self):
        return None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # neuron 参数 fp32, 矩阵 bf16: 用 autocast 让 forward 里的 matmul 自动匹配.
        # (用户 AutoModel.from_pretrained 后直接 model(ids) 就能跑, 无需手动 autocast)
        device_type = 'cuda' if (input_ids is not None and input_ids.is_cuda) else 'cpu'
        with torch.amp.autocast(device_type, dtype=torch.bfloat16):
            out = self.snn(input_ids)
        logits = out.logits
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1), ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids=None, max_new_tokens=256, temperature=1.0,
                 top_k=50, top_p=1.0, repetition_penalty=1.0,
                 eos_token_id=None, pad_token_id=None,
                 attention_mask=None, **kwargs):
        """覆写 generate: 委托给 SNNLanguageModel.generate (stateful SNN 生成).

        HF 默认 generate 每步调 forward() → SNNLanguageModel.forward() 内部
        functional.reset_net() 清状态 → 每步丢历史 → 乱码.
        SNNLanguageModel.generate 只在 prompt prefill 前 reset 一次,
        之后跨 token 保持 PLIF .v 状态.

        temperature 按用户传入透传, 不做 do_sample 自动重映射 (用户要 greedy 请传 temperature<=0).
        """
        if input_ids is None:
            raise ValueError('input_ids required')
        device_type = 'cuda' if input_ids.is_cuda else 'cpu'
        with torch.amp.autocast(device_type, dtype=torch.bfloat16):
            out = self.snn.generate(
                prompt_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k else 0,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
            )
        return out

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """覆写 from_pretrained 强制全量 fp32 加载 + 重算 RoPE 非持久 buffer.

        1) dtype: safetensors 中 neuron 是 fp32 / 矩阵是 bf16. 若直接用 bf16
           加载会把 neuron 降 bf16 (精度不足); 若混合 cast 又和 model.py
           native (upcast 一切到 fp32) 不一致, argmax 飘. fp32 加载 + forward
           内 autocast bf16 compute, 与原生 bit-exact 对齐.

        2) RoPE buffer: SNNAttentionDecoderLayer 的 rope_cos/rope_sin 用
           register_buffer(..., persistent=False), 不在 safetensors 里.
           transformers 5.x 的 fast-init / meta-device 路径跳过 __init__
           的 _precompute_rope_freqs 计算 → buffer 是未初始化内存 → 结果乱.
           load 完后重新调用 _precompute_rope_freqs 重填.
        """
        import torch as _torch
        user_dtype = kwargs.get('dtype', kwargs.get('torch_dtype', None))
        if user_dtype is None:
            kwargs['dtype'] = _torch.float32
        model = super().from_pretrained(*args, **kwargs)
        # 重填 RoPE non-persistent buffer
        # _precompute_rope_freqs(dim) 返回 (max_seq_len, dim//2), 所以 dim = shape[-1] * 2
        for layer_mod in model.snn.layers:
            if hasattr(layer_mod, 'rope_cos') and hasattr(layer_mod, 'rope_sin'):
                dim = layer_mod.rope_cos.shape[-1] * 2
                cos, sin = _precompute_rope_freqs(dim)
                device = layer_mod.rope_cos.device
                dtype = layer_mod.rope_cos.dtype
                layer_mod.rope_cos.data = cos.to(device=device, dtype=dtype)
                layer_mod.rope_sin.data = sin.to(device=device, dtype=dtype)
        return model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def can_generate(self):
        return True

    _SENTINEL = object()

    @torch.no_grad()
    def generate(self, input_ids=None, max_new_tokens=_SENTINEL,
                 temperature=_SENTINEL, top_k=_SENTINEL, top_p=_SENTINEL,
                 repetition_penalty=_SENTINEL, eos_token_id=_SENTINEL, **kwargs):
        defaults = dict(max_new_tokens=256, temperature=1.0, top_k=50,
                        top_p=1.0, repetition_penalty=1.0, eos_token_id=None)
        gen_config = kwargs.get('generation_config', None)
        if gen_config is not None:
            for key in defaults:
                v = getattr(gen_config, key, None)
                if v is not None:
                    defaults[key] = v
            if not getattr(gen_config, 'do_sample', True):
                defaults['temperature'] = 0.0
        S = self._SENTINEL
        if max_new_tokens is not S: defaults['max_new_tokens'] = max_new_tokens
        if temperature is not S: defaults['temperature'] = temperature
        if top_k is not S: defaults['top_k'] = top_k
        if top_p is not S: defaults['top_p'] = top_p
        if repetition_penalty is not S: defaults['repetition_penalty'] = repetition_penalty
        if eos_token_id is not S: defaults['eos_token_id'] = eos_token_id
        if not kwargs.get('do_sample', True):
            defaults['temperature'] = 0.0
        if 'max_length' in kwargs and input_ids is not None:
            derived = kwargs['max_length'] - input_ids.shape[1]
            if derived <= 0:
                return input_ids
            defaults['max_new_tokens'] = derived
        if kwargs.get('num_beams', 1) != 1:
            raise NotImplementedError("NeuronSpark SNN does not support beam search")
        if kwargs.get('num_return_sequences', 1) != 1:
            raise NotImplementedError("NeuronSpark SNN does not support multiple return sequences")
        if 'attention_mask' in kwargs:
            mask = kwargs['attention_mask']
            if mask is not None and mask.min() == 0:
                raise ValueError("NeuronSpark SNN generate does not support padding in attention_mask")
        if defaults['eos_token_id'] is None:
            defaults['eos_token_id'] = self.config.eos_token_id
        return self.snn.generate(
                input_ids, max_new_tokens=defaults['max_new_tokens'],
                temperature=defaults['temperature'], top_k=defaults['top_k'],
                top_p=defaults['top_p'], repetition_penalty=defaults['repetition_penalty'],
                eos_token_id=defaults['eos_token_id'],
            )

