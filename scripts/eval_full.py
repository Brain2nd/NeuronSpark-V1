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
        # 自动检测 HF artifact vs native. HF 走 AutoModelForCausalLM (含 RoPE buffer 重算 + fp32 load),
        # native 走 load_model_weights. 二者对同一 ckpt 应产出 bit-exact 相同模型.
        import os, json
        is_hf = False
        cfg_path = os.path.join(checkpoint, 'config.json')
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                _cfg = json.load(f)
            is_hf = 'auto_map' in _cfg or 'architectures' in _cfg
        if is_hf:
            from transformers import AutoModelForCausalLM as _AMLM
            self.model = _AMLM.from_pretrained(
                checkpoint, trust_remote_code=True,
            ).to(self._device).eval()
        else:
            snn_config = load_config(checkpoint)
            config = NeuronSparkConfig(**snn_config)
            self.model = NeuronSparkForCausalLM(config)
            load_model_weights(checkpoint, self.model.snn, str(self._device))
            self.model = self.model.to(self._device).eval()
        # 混合精度统一 (neuron fp32, 其余 bf16): autocast 负责 compute
        for name, param in self.model.named_parameters():
            if name.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th')):
                param.data = param.data.float()
            else:
                param.data = param.data.to(self._dtype)
        for _, buf in self.model.named_buffers():
            buf.data = buf.data.to(self._dtype)

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

    # --- chat template 支持: lm-eval 传 apply_chat_template=True 时调用 ---
    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, 'name_or_path', 'neuronspark-tokenizer')

    def chat_template(self, chat_template: bool | str = False) -> str:
        """返回 tokenizer 的 jinja2 chat_template 字符串 (用于 cache key)."""
        tpl = getattr(self.tokenizer, 'chat_template', None)
        return tpl or ''

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

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
            # lm-eval 新版 loglikelihood_rolling 期望纯 float (wikitext metric weighted_mean
            # 依赖于此), 不能像 loglikelihood 那样返回 tuple.
            results.append(total_ll)
        return results

    def generate_until(self, requests):
        return [''] * len(requests)


def run_eval(checkpoint, device, tasks, output_path, apply_chat_template=False,
             system_instruction=None):
    """单 GPU 测评指定任务子集。"""
    # 在多进程下, 每个子进程继承 parent 的 current_device=cuda:0. 必须显式 set_device
    # 否则 Triton kernel 或其它中间 tensor 会落在 cuda:0 造成跨设备/CPU 假象.
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))
    print(f'[{device}] Running {len(tasks)} tasks on {checkpoint} '
          f'(chat_template={apply_chat_template})...')
    results = lm_eval.simple_evaluate(
        model='neuronspark',
        model_args=f'checkpoint={checkpoint},device={device}',
        tasks=tasks,
        batch_size=1,
        apply_chat_template=apply_chat_template,
        system_instruction=system_instruction,
    )

    for task, res in results['results'].items():
        acc = res.get('acc,none', res.get('acc', '?'))
        acc_n = res.get('acc_norm,none', '')
        norm_str = f' (norm: {acc_n:.4f})' if isinstance(acc_n, float) else ''
        if isinstance(acc, float):
            print(f'  [{device}] {task:>20s}: {acc:.4f}{norm_str}')
        else:
            print(f'  [{device}] {task:>20s}: {acc}')

    os.makedirs('exp', exist_ok=True)
    out = {
        'checkpoint': os.path.basename(checkpoint),
        'device': device,
        'tasks': tasks,
        'results': {
            t: {k: v for k, v in r.items() if not k.startswith('samples')}
            for t, r in results['results'].items()
        },
    }
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[{device}] Saved to {output_path}')
    return out


def merge_results(partial_files, merged_path):
    """合并多个 partial JSON 为一个完整结果文件。"""
    os.makedirs(os.path.dirname(merged_path) or '.', exist_ok=True)
    merged = {'results': {}, 'tasks': []}
    for pf in partial_files:
        if not os.path.exists(pf):
            continue
        with open(pf) as f:
            part = json.load(f)
        merged['results'].update(part['results'])
        merged['tasks'].extend(part['tasks'])
        merged['checkpoint'] = part.get('checkpoint', '')
    with open(merged_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'Merged → {merged_path}')
    return merged


if __name__ == '__main__':
    import argparse, multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ckpt_step441000')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--apply_chat_template', action='store_true',
                        help='按 SFT ChatML 格式包装 prompt (要求 LM 实现 apply_chat_template)')
    parser.add_argument('--system_instruction', type=str, default=None,
                        help='apply_chat_template 时的 system prompt. 默认 None — 对齐 SFT '
                             '训练数据 (benchmark_sft_mix 90000 条样本均无 system message). '
                             '显式传该参数只在训练时也带 system 的场景下有意义.')
    parser.add_argument('--tag', type=str, default='',
                        help='输出文件名额外后缀, 避免覆盖 (如 chat)')
    args = parser.parse_args()

    # Task suite (ALL models go through these by default, 2026-04-21 起扩充):
    #   9 main multi-choice/knowledge tasks (zero-shot loglik):
    #     arc_easy / arc_challenge / hellaswag / winogrande / boolq / mmlu / piqa / openbookqa / ceval-valid
    #   4 Phase A raw-LM probes:
    #     wikitext (word perplexity via loglikelihood_rolling), sst2, mnli, xnli_zh
    # 已永久移除:
    #   lambada_openai — chat/SFT 模型任务形式错配 (ppl 爆 1M+), 不再评测
    # 可选单独加 (默认不启用):
    #   blimp — 67 英文 subtask, 我们的 torch.compile PLIF kernel 在变长输入上 cache miss 会 hang
    #     若需评测, 手动传 --tasks blimp_adjunct_island blimp_... (avoiding blimp_nl_* 荷兰语变体)
    all_tasks = [
        'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'boolq',
        'mmlu', 'piqa', 'openbookqa', 'ceval-valid',
        'wikitext', 'sst2', 'mnli', 'xnli_zh',
    ]

    ckpt_name = os.path.basename(args.checkpoint)
    tag = f'_{args.tag}' if args.tag else ''
    output_path = args.output or f'exp/lm_eval_full_{ckpt_name}{tag}.json'

    if args.num_gpus <= 1:
        run_eval(args.checkpoint, 'cuda:0', all_tasks, output_path,
                 apply_chat_template=args.apply_chat_template,
                 system_instruction=args.system_instruction)
    else:
        # 按请求量均衡分配, 1 GPU 1 重任务原则: mmlu / hellaswag / mnli 各独占一卡.
        # 支持 4 / 8 GPU 两种布局 (H100 8 卡 / 4090 4 卡).
        if args.num_gpus >= 8:
            task_groups = [
                ['mmlu'],                               # GPU 0 (~56K reqs)
                ['hellaswag'],                          # GPU 1 (~40K)
                ['mnli'],                               # GPU 2 (~30K)
                ['xnli_zh', 'arc_easy'],                # GPU 3 (~17K)
                ['arc_challenge', 'boolq'],             # GPU 4 (~11K)
                ['ceval-valid', 'piqa'],                # GPU 5 (~9K)
                ['openbookqa', 'winogrande', 'sst2'],   # GPU 6 (~6K)
                ['wikitext'],                           # GPU 7 (tiny)
            ]
        else:  # 4 GPU
            task_groups = [
                ['mmlu'],                                                                      # GPU 0 (~14K)
                ['hellaswag'],                                                                 # GPU 1 (~10K)
                ['mnli', 'sst2', 'xnli_zh'],                                                   # GPU 2 (~13K)
                ['ceval-valid', 'arc_easy', 'arc_challenge', 'winogrande', 'boolq',
                 'piqa', 'openbookqa', 'wikitext'],                                            # GPU 3 (~11K)
            ]
        # 如果 GPU 数不足 4，合并剩余任务到最后一组
        while len(task_groups) > args.num_gpus:
            task_groups[-2].extend(task_groups.pop())

        partial_files = []
        procs = []
        for i, tasks in enumerate(task_groups):
            device = f'cuda:{i}'
            pf = f'exp/lm_eval_{ckpt_name}{tag}_gpu{i}.json'
            partial_files.append(pf)
            p = multiprocessing.Process(
                target=run_eval,
                args=(args.checkpoint, device, tasks, pf,
                      args.apply_chat_template, args.system_instruction),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # 合并
        merged = merge_results(partial_files, output_path)

        # 打印汇总
        print(f'\n{"="*60}')
        print(f'RESULTS: {ckpt_name}')
        print(f'{"="*60}')
        for task, res in merged['results'].items():
            acc = res.get('acc,none', res.get('acc', '?'))
            acc_n = res.get('acc_norm,none', '')
            norm_str = f' (norm: {acc_n:.4f})' if isinstance(acc_n, float) else ''
            if isinstance(acc, float):
                print(f'  {task:>20s}: {acc:.4f}{norm_str}')
            else:
                print(f'  {task:>20s}: {acc}')
