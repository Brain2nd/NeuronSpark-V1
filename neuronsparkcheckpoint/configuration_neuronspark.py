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
        v_th_min=0.1,
        memory_layer_interval=4,
        D_key=128,
        D_value=128,
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
