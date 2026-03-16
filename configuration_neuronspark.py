"""NeuronSpark 模型配置。"""

from transformers import PretrainedConfig


class NeuronSparkConfig(PretrainedConfig):
    """
    SNN 隐状态空间语言模型配置。

    Args:
        vocab_size: 词表大小
        D: 隐层维度
        N: 状态扩展因子（每通道隐神经元数）
        K: 每 token 最大 SNN 时间步（PonderNet 动态决定有效步数）
        num_layers: SNN 解码层数
        D_ff: FFN 中间层维度
        v_th_min: 动态阈值下限
    """
    model_type = "neuronspark"

    def __init__(
        self,
        vocab_size=6144,
        D=896,
        N=8,
        K=16,
        num_layers=20,
        D_ff=2688,
        v_th_min=0.1,
        **kwargs,
    ):
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff
        self.v_th_min = v_th_min
        super().__init__(vocab_size=vocab_size, **kwargs)
