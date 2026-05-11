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
        # 神经元发放形式 (v4.1 — 见 docs/v4_status_and_roadmap.md §神经元设计)
        #   "supra"   = bio-ReLU 超阈电流: output = relu(V_pre - v_th), V_post = min(V_pre, v_th) — 精确 ReLU 梯度
        #   "quantal" = 量子化释放: output = v_th·𝟙[V_pre>v_th], V_post = V_pre - v_th·𝟙[...] (剩余余量留膜里) — surrogate 梯度
        spike_mode="supra",
        # surrogate gradient α (sigmoid surrogate, 仅 spike_mode="quantal" 时用于 output 的反向)
        surrogate_alpha=4.0,
        # 后超极化 (AHP / 不应期): 发放后膜额外下压 ahp (per-channel 可学), V_post -= ahp·𝟙[V_pre>v_th]
        use_ahp=False,
        ahp_init=0.0,  # ahp 参数初始值 (per-channel scalar)
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
        # HF GenerationMixin / DynamicCache 期望 num_hidden_layers 字段
        self.num_hidden_layers = num_layers
        # SNN 没有 KV cache, 关掉避免 HF 试图建 DynamicCache
        self.use_cache = False
        self.D_ff = D_ff
        self.v_th_min = v_th_min
        self.memory_layer_interval = memory_layer_interval
        self.D_key = D_key
        self.D_value = D_value
        self.spike_mode = spike_mode
        self.surrogate_alpha = surrogate_alpha
        self.use_ahp = use_ahp
        self.ahp_init = ahp_init
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
