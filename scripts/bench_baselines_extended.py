"""扩展基线测评: 补齐 mmlu/piqa/openbookqa/lambada_openai/ceval-valid 5 项.

绕开 lm-eval HFLM (会硬塞 dtype kwarg 导致新 transformers 报错),
自定义 LM 子类直接用 AutoModelForCausalLM.from_pretrained 加载.
"""

import argparse
import json
import os
import datetime
import gc
import torch
import torch.nn.functional as F
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "pythia-1b": "EleutherAI/pythia-1b",
    "opt-1.3b": "facebook/opt-1.3b",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base",
    "mamba-790m": "state-spaces/mamba-790m-hf",
    "mamba-1.4b": "state-spaces/mamba-1.4b-hf",
}

DEFAULT_EXT_TASKS = ["openbookqa", "piqa", "ceval-valid", "mmlu"]  # lambada_openai 已移除 (chat 模型不适用)


@register_model("hf_manual")
class HFManualLM(LM):
    """手动装载 HF 模型的 lm-eval wrapper.
    绕开 HFLM 内置 dtype kwarg 与新 transformers 不兼容.
    """
    def __init__(self, pretrained: str, device: str = "cuda:0",
                 dtype: str = "bfloat16", max_length: int = 2048, **kw):
        super().__init__()
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)
        self._max_length = int(max_length)
        torch.cuda.set_device(self._device)
        print(f"  [hf_manual] loading {pretrained} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device).eval()

    @property
    def eot_token_id(self): return self.tokenizer.eos_token_id
    @property
    def max_length(self): return self._max_length
    @property
    def max_gen_toks(self): return 256
    @property
    def batch_size(self): return 1
    @property
    def device(self): return self._device
    def tok_encode(self, s, **kw): return self.tokenizer.encode(s, add_special_tokens=False)
    def tok_decode(self, t, **kw): return self.tokenizer.decode(t)

    def loglikelihood(self, requests):
        out = []
        for req in requests:
            ctx, cont = req.args[0], req.args[1]
            ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(cont, add_special_tokens=False)
            all_ids = ctx_ids + cont_ids
            if len(all_ids) > self.max_length:
                all_ids = all_ids[-self.max_length:]
            x = torch.tensor([all_ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                o = self.model(input_ids=x)
            logits = o.logits[0, :-1, :]
            labels = x[0, 1:]
            lp = F.log_softmax(logits.float(), dim=-1)
            cs = len(all_ids) - len(cont_ids) - 1
            ll = 0.0
            greedy = True
            for i in range(len(cont_ids)):
                pos = cs + i
                if 0 <= pos < lp.shape[0]:
                    tid = labels[pos].item()
                    ll += lp[pos, tid].item()
                    if lp[pos].argmax().item() != tid:
                        greedy = False
            out.append((ll, greedy))
        return out

    def loglikelihood_rolling(self, requests):
        out = []
        for req in requests:
            ids = self.tokenizer.encode(req.args[0], add_special_tokens=False)
            if len(ids) > self.max_length:
                ids = ids[-self.max_length:]
            x = torch.tensor([ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                o = self.model(input_ids=x)
            logits = o.logits[0, :-1, :]
            labels = x[0, 1:]
            lp = F.log_softmax(logits.float(), dim=-1)
            ll = sum(lp[i, labels[i].item()].item() for i in range(labels.shape[0]))
            out.append((ll,))
        return out

    def generate_until(self, requests):
        return [''] * len(requests)


def run_eval(model_name, hf_id, tasks, device, tag):
    print(f"\n{'=' * 60}")
    print(f"[{device}] Evaluating {model_name} ({hf_id}) on {tasks}")
    print(f"{'=' * 60}")

    results = lm_eval.simple_evaluate(
        model="hf_manual",
        model_args=f"pretrained={hf_id},device={device},dtype=bfloat16",
        tasks=tasks,
        batch_size=1,
    )

    summary = {
        t: {k: v for k, v in r.items() if not k.startswith("samples")}
        for t, r in results["results"].items()
    }

    for task, m in summary.items():
        acc = m.get("acc,none", m.get("acc", "?"))
        acc_n = m.get("acc_norm,none", "")
        ns = f" (norm: {acc_n:.4f})" if isinstance(acc_n, float) else ""
        if isinstance(acc, float):
            print(f"  {task:>20s}: {acc:.4f}{ns}")

    os.makedirs("exp", exist_ok=True)
    path = f"exp/lm_eval_baseline_{model_name}_{tag}.json"
    with open(path, "w") as f:
        json.dump({
            "model": model_name, "hf_id": hf_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "tasks": tasks, "device": device, "results": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {path}")

    del results
    gc.collect()
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_EXT_TASKS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tag", type=str, default="ext")
    args = parser.parse_args()

    selected = {k: MODELS[k] for k in (args.models or MODELS.keys()) if k in MODELS}
    for name, hf_id in selected.items():
        try:
            run_eval(name, hf_id, args.tasks, args.device, args.tag)
        except Exception as e:
            print(f"  ERROR: {name} failed: {type(e).__name__}: {e}")
