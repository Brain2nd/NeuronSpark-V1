from transformers import PretrainedConfig


class NeuronSparkConfig(PretrainedConfig):
    model_type = "neuronspark"

    def __init__(
        self,
        vocab_size=64002,
        D=1024,
        N=8,
        K=12,
        num_layers=24,
        D_ff=3072,
        v_th_min=0.02,  # bio-ReLU: 下调阈值下限提升初始发放率至健康区 30-50%
        memory_layer_interval=4,
        D_key=128,
        D_value=128,
        # v3 PonderNet fields (input-conditioned KPredictor)
        k_predictor_hidden=None,
        ponder_T_init=2.0,
        ponder_T_final=0.3,
        eps_explore=0.05,
        bias_balancing_lr=1e-3,
        bias_balancing_ema=0.99,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff
        self.v_th_min = v_th_min
        self.memory_layer_interval = memory_layer_interval
        self.D_key = D_key
        self.D_value = D_value
        # v3 PonderNet
        self.k_predictor_hidden = k_predictor_hidden
        self.ponder_T_init = ponder_T_init
        self.ponder_T_final = ponder_T_final
        self.eps_explore = eps_explore
        self.bias_balancing_lr = bias_balancing_lr
        self.bias_balancing_ema = bias_balancing_ema

        # auto_map: HF 文件路径/类名 两段式（neuronspark/ 子目录）
        kwargs.setdefault("auto_map", {
            "AutoConfig": "configuration_neuronspark.NeuronSparkConfig",
            "AutoModelForCausalLM": "modeling_neuronspark.NeuronSparkForCausalLM",
        })
        kwargs.setdefault("architectures", ["NeuronSparkForCausalLM"])
        kwargs.setdefault("dtype", "bfloat16")

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
