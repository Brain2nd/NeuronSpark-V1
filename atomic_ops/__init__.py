import torch


class _SpikeCurrentFn(torch.autograd.Function):
    """融合脉冲电流激活：前向 = v_th * spike，反向梯度走稠密路径。

    原版用 detach 技巧创建 ~5 个中间 tensor（sc, dense, sc-dense 等），
    每个与输入同形状，在 DN=8192 时每个 ~1 GiB（bf16），严重浪费显存。
    本版仅保存 spike 和 v_th 两个 tensor，节省 ~60% 激活显存。
    """

    @staticmethod
    def forward(ctx, spike, v_th, eps):
        ctx.save_for_backward(spike, v_th)
        ctx.eps = eps
        return v_th * spike

    @staticmethod
    def backward(ctx, grad_output):
        spike, v_th = ctx.saved_tensors
        eps = ctx.eps
        # 原版 dense = v_th*spike + eps*(1-spike)，梯度走 dense 路径：
        # ∂dense/∂spike = v_th - eps
        grad_spike = grad_output * (v_th - eps)
        # ∂dense/∂v_th = spike
        grad_v_th = grad_output * spike
        # v_th 可能有 broadcast（如 shape (1, batch, D) 对 (TK, batch, D)），需 reduce
        if grad_v_th.shape != v_th.shape:
            reduce_dims = tuple(
                i for i in range(grad_v_th.dim())
                if v_th.shape[i] == 1 and grad_v_th.shape[i] > 1
            )
            if reduce_dims:
                grad_v_th = grad_v_th.sum(dim=reduce_dims, keepdim=True)
        return grad_spike, grad_v_th, None


def spike_current_activation(spike, v_th, eps=1e-6):
    """脉冲电流激活：前向 = V_th * spike（稀疏），反向梯度稠密。"""
    return _SpikeCurrentFn.apply(spike, v_th, eps)


from .selective_plif import SelectivePLIFNode
from .plif_node import PLIFNode
from .lateral_inhibition import LateralInhibition
from .snn_block import SNNBlock
from .snn_ffn import SNNFFN
from .snn_decoder_layer import SNNDecoderLayer
from .parallel_scan import hillis_steele_scan, linear_recurrence, plif_parallel_forward
from .fp16_codec import fp16_encode, fp16_decode
from .rms_norm import RMSNorm
