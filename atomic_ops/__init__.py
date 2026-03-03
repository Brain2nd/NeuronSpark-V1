import torch


def spike_current_activation(spike, v_th, eps=1e-6):
    """脉冲电流激活：前向 = V_th * spike（稀疏），反向梯度稠密。"""
    sc = v_th * spike
    dense = sc + eps * (1.0 - spike)
    return dense + (sc - dense).detach()


from .selective_plif import SelectivePLIFNode
from .plif_node import PLIFNode
from .lateral_inhibition import LateralInhibition
from .snn_block import SNNBlock
from .snn_ffn import SNNFFN
from .snn_decoder_layer import SNNDecoderLayer
from .parallel_scan import hillis_steele_scan, linear_recurrence, plif_parallel_forward
from .fp16_codec import fp16_encode, fp16_decode
from .rms_norm import RMSNorm
