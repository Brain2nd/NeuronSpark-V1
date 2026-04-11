"""
lm-eval-harness 自定义模型注册：NeuronSpark SNN。

用法:
    lm_eval --model neuronspark \
            --model_args checkpoint=checkpoints/ckpt_step257000,tokenizer_path=./tokenizer/ \
            --tasks sst2,hellaswag \
            --batch_size 1
"""

import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from checkpoint_utils import load_config, load_model_weights


@register_model("neuronspark")
class NeuronSparkLM(LM):

    def __init__(
        self,
        checkpoint="checkpoints/ckpt_step257000",
        tokenizer_path="./tokenizer/",
        device="cuda:0",
        batch_size=1,
        dtype="bfloat16",
        **kwargs,
    ):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        snn_config = load_config(checkpoint)
        config = NeuronSparkConfig(**snn_config)
        self.model = NeuronSparkForCausalLM(config)
        load_model_weights(checkpoint, self.model.snn, str(self._device))
        self.model = self.model.to(device=self._device, dtype=self._dtype).eval()

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        results = []
        for context, continuation in requests:
            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            all_ids = ctx_ids + cont_ids

            if len(all_ids) > self.max_length:
                all_ids = all_ids[-self.max_length:]
                cont_len = len(cont_ids)
            else:
                cont_len = len(cont_ids)

            input_ids = torch.tensor([all_ids], dtype=torch.long, device=self._device)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=self._dtype):
                out = self.model(input_ids=input_ids)

            # logits shape: (1, seq_len, vocab)
            # shift: logits[t] predicts token[t+1]
            shift_logits = out.logits[0, :-1, :]  # (seq_len-1, vocab)
            shift_labels = input_ids[0, 1:]  # (seq_len-1,)

            log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)

            # continuation 对应的位置: 最后 cont_len 个 token
            cont_start = len(all_ids) - cont_len - 1  # shift 后的起始位置
            cont_log_probs = []
            for i in range(cont_len):
                pos = cont_start + i
                if pos < 0 or pos >= log_probs.shape[0]:
                    continue
                token_id = shift_labels[pos].item()
                cont_log_probs.append(log_probs[pos, token_id].item())

            total_ll = sum(cont_log_probs)
            is_greedy = True
            if cont_log_probs:
                for i in range(cont_len):
                    pos = cont_start + i
                    if pos < 0 or pos >= log_probs.shape[0]:
                        continue
                    if log_probs[pos].argmax().item() != shift_labels[pos].item():
                        is_greedy = False
                        break

            results.append((total_ll, is_greedy))

        return results

    def loglikelihood(self, requests):
        new_requests = []
        for req in requests:
            new_requests.append((req.args[0], req.args[1]))
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            text = req.args[0]
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[-self.max_length:]

            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=self._dtype):
                out = self.model(input_ids=input_ids)

            shift_logits = out.logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)

            total_ll = 0.0
            for i in range(shift_labels.shape[0]):
                total_ll += log_probs[i, shift_labels[i].item()].item()

            results.append((total_ll,))

        return results

    def generate_until(self, requests):
        results = []
        for req in requests:
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self._device)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=self._dtype):
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen,
                    temperature=0.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            gen_text = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )

            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text[:gen_text.index(stop)]
                    break

            results.append(gen_text)

        return results
