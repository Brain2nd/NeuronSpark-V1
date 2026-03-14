import torch


class _SpikeCurrentFn(torch.autograd.Function):
    """融合脉冲电流激活：前向 = v_th * spike，反向梯度走稠密路径。

    显存优化: spike ∈ {0,1} 存为 uint8（1 字节/元素），
    backward 时 cast 回计算 dtype，节省 50% spike 显存。
    """

    @staticmethod
    def forward(ctx, spike, v_th, eps):
        # spike ∈ {0,1}: 存为 uint8 节省 50% 显存
        ctx.save_for_backward(spike.to(torch.uint8), v_th)
        ctx.eps = eps
        ctx._spike_dtype = spike.dtype
        return v_th * spike

    @staticmethod
    def backward(ctx, grad_output):
        spike_u8, v_th = ctx.saved_tensors
        spike = spike_u8.to(ctx._spike_dtype)  # uint8 → 计算 dtype
        eps = ctx.eps
        # ∂dense/∂spike = v_th - eps
        grad_spike = grad_output * (v_th - eps)
        # ∂sc/∂v_th = +spike：spike=1 时 output=V_th，增大 V_th 直接增大输出
        # spike=0 时 grad=0 无影响；阈值调节由 surrogate gradient 经 spike 路径处理
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
from .hyper_connection import HyperConnection, sinkhorn_log, sinkhorn_projection
from .parallel_scan import hillis_steele_scan, linear_recurrence, plif_parallel_forward
from .fp16_codec import fp16_encode, fp16_decode
from .rms_norm import RMSNorm

# v3 组件
from .bio_ssm_layer import BioSSMLayer
from .bio_latent_moe import BioLatentMoELayer
from .bio_attention import BioAttentionLayer
from .mtp_head import MTPHead
