"""
NeuronSpark: SNN 隐状态空间语言模型 — HuggingFace 接口

用法:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints_sft/", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("checkpoints_sft/")
"""

from typing import Optional

import torch
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_neuronspark import NeuronSparkConfig
from model import SNNLanguageModel


class NeuronSparkForCausalLM(PreTrainedModel, GenerationMixin):
    """
    SNN 语言模型 — CausalLM 接口。

    封装 SNNLanguageModel，提供 HuggingFace 标准接口:
      - forward(input_ids, labels) → CausalLMOutputWithPast
      - generate() 支持（通过 GenerationMixin）
    """
    config_class = NeuronSparkConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: NeuronSparkConfig):
        super().__init__(config)
        self.model = SNNLanguageModel(
            vocab_size=config.vocab_size,
            D=config.D,
            N=config.N,
            K=config.K,
            num_layers=config.num_layers,
            D_ff=config.D_ff,
            v_th_min=config.v_th_min,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        # tied head: 输出复用 embed_tokens.weight
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        前向传播。

        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) 目标 token IDs（可选，用于计算 loss）
            attention_mask: 兼容参数（SNN 无 attention，忽略）
        """
        if labels is not None:
            out = self.model(input_ids, target_ids=labels)
            # 计算 masked loss
            loss_mask = (labels != 0).float().view(-1)
            loss = (out.last_loss * loss_mask).sum() / loss_mask.sum()
            # 加 ponder cost
            if out.ponder_cost is not None:
                loss = loss + 0.01 * out.ponder_cost
            return CausalLMOutputWithPast(loss=loss)
        else:
            out = self.model(input_ids)
            return CausalLMOutputWithPast(logits=out.logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """generate() 所需的输入准备。"""
        return {"input_ids": input_ids}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        自回归生成（直接调用 SNN 的 generate 方法）。
        """
        return self.model.generate(
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
