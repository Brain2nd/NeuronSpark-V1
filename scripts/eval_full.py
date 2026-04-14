"""扩展测评: 10 个 benchmark 任务。"""
import sys, os, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from checkpoint_utils import load_config, load_model_weights
from transformers import AutoTokenizer
import torch.nn.functional as F
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model('neuronspark')
class NeuronSparkLM(LM):
    def __init__(self, checkpoint='checkpoints/ckpt_step441000',
                 tokenizer_path='./tokenizer/', device='cuda:0',
                 batch_size=1, dtype='bfloat16', **kw):
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
    def eot_token_id(self): return self.tokenizer.eos_token_id
    @property
    def max_length(self): return 2048
    @property
    def max_gen_toks(self): return 256
    @property
    def batch_size(self): return self._batch_size
    @property
    def device(self): return self._device
    def tok_encode(self, s, **kw): return self.tokenizer.encode(s, add_special_tokens=False)
    def tok_decode(self, t, **kw): return self.tokenizer.decode(t)

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            ctx, cont = req.args[0], req.args[1]
            ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(cont, add_special_tokens=False)
            all_ids = ctx_ids + cont_ids
            if len(all_ids) > self.max_length:
                all_ids = all_ids[-self.max_length:]
            cont_len = len(cont_ids)
            input_ids = torch.tensor([all_ids], dtype=torch.long, device=self._device)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=self._dtype):
                out = self.model(input_ids=input_ids)
            shift_logits = out.logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            cont_start = len(all_ids) - cont_len - 1
            total_ll = 0.0
            is_greedy = True
            for i in range(cont_len):
                pos = cont_start + i
                if pos < 0 or pos >= log_probs.shape[0]:
                    continue
                tid = shift_labels[pos].item()
                total_ll += log_probs[pos, tid].item()
                if log_probs[pos].argmax().item() != tid:
                    is_greedy = False
            results.append((total_ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            token_ids = self.tokenizer.encode(req.args[0], add_special_tokens=False)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[-self.max_length:]
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=self._dtype):
                out = self.model(input_ids=input_ids)
            shift_logits = out.logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            total_ll = sum(log_probs[i, shift_labels[i].item()].item()
                          for i in range(shift_labels.shape[0]))
            results.append((total_ll,))
        return results

    def generate_until(self, requests):
        return [''] * len(requests)


if __name__ == '__main__':
    tasks = [
        'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'boolq',
        'mmlu', 'piqa', 'openbookqa', 'lambada_openai',
        'ceval-valid',
    ]
    print(f'Running {len(tasks)}-task eval on step 428000...')
    results = lm_eval.simple_evaluate(model='neuronspark', tasks=tasks, batch_size=1)

    for task, res in results['results'].items():
        acc = res.get('acc,none', res.get('acc', '?'))
        acc_n = res.get('acc_norm,none', '')
        norm_str = f' (norm: {acc_n:.4f})' if isinstance(acc_n, float) else ''
        if isinstance(acc, float):
            print(f'  {task:>20s}: {acc:.4f}{norm_str}')
        else:
            print(f'  {task:>20s}: {acc}')

    os.makedirs('exp', exist_ok=True)
    out = {
        'checkpoint': 'ckpt_step441000',
        'tasks': tasks,
        'results': {
            t: {k: v for k, v in r.items() if not k.startswith('samples')}
            for t, r in results['results'].items()
        },
    }
    with open('exp/lm_eval_full_ckpt_step441000.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved to exp/lm_eval_full_ckpt_step441000.json')
