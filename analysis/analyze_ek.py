"""
生物可解释性分析：PonderNet E[K] vs 语言学特征

分析内容：
1. 每个 token 的 E[K]（每层每子层的期望 SNN 步数）
2. E[K] vs 词性（名词/动词/标点/停用词）
3. E[K] vs surprisal（-log P(token)）
4. E[K] vs 词频
5. 逐层 firing rate
6. β 分布

用法:
    python analysis/analyze_ek.py
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional

from model import SNNLanguageModel


# ============================================================
# 1. 模型加载 + Hook 注入
# ============================================================

def load_model(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('model_config', {})
    model = SNNLanguageModel(
        vocab_size=config.get('vocab_size', 6144),
        D=config.get('D', 1024), N=config.get('N', 8), K=config.get('K', 32),
        num_layers=config.get('num_layers', 20), D_ff=config.get('D_ff', 3072),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()
    print(f"Loaded model: D={config.get('D')}, K={config.get('K')}, L={config.get('num_layers')}")
    return model, config


class EKCollector:
    """Hook 收集每层每子层的 per-token E[K]。"""

    def __init__(self, model):
        self.model = model
        self.ek_data = {}  # {layer_idx: {'block': (seq_len, batch), 'ffn': (seq_len, batch)}}
        self._hooks = []
        self._install_hooks()

    def _install_hooks(self):
        """Monkey-patch _adaptive_aggregate 以捕获 expected_k。"""
        for layer_idx, layer in enumerate(self.model.layers):
            original_fn = layer._adaptive_aggregate

            def make_hook(idx, orig):
                call_count = [0]  # 每次 forward_parallel 调用两次 (block + ffn)

                def hooked(frames, halt_proj):
                    aggregated, ponder_cost, expected_k = orig(frames, halt_proj)
                    sublayer = 'block' if call_count[0] % 2 == 0 else 'ffn'
                    if idx not in self.ek_data:
                        self.ek_data[idx] = {}
                    self.ek_data[idx][sublayer] = expected_k.cpu().numpy()  # (seq_len, batch)
                    call_count[0] += 1
                    return aggregated, ponder_cost, expected_k

                return hooked

            layer._adaptive_aggregate = make_hook(layer_idx, original_fn)

    def reset(self):
        self.ek_data = {}

    def get_ek_per_token(self):
        """返回每个 token 在所有层的平均 E[K]。shape: (seq_len,)"""
        all_ek = []
        for layer_idx in sorted(self.ek_data.keys()):
            for sublayer in ['block', 'ffn']:
                if sublayer in self.ek_data[layer_idx]:
                    ek = self.ek_data[layer_idx][sublayer]  # (seq_len, batch)
                    all_ek.append(ek[:, 0])  # batch=0
        if not all_ek:
            return None
        return np.mean(all_ek, axis=0)  # (seq_len,) 所有层平均

    def get_ek_per_layer(self):
        """返回每层的 E[K]。shape: (num_layers, 2, seq_len)"""
        result = {}
        for layer_idx in sorted(self.ek_data.keys()):
            result[layer_idx] = {}
            for sublayer in ['block', 'ffn']:
                if sublayer in self.ek_data[layer_idx]:
                    result[layer_idx][sublayer] = self.ek_data[layer_idx][sublayer][:, 0]
        return result


# ============================================================
# 2. 中文词性标注（jieba）
# ============================================================

def get_pos_tags(text):
    """用 jieba 做中文词性标注。"""
    try:
        import jieba.posseg as pseg
        words = list(pseg.cut(text))
        return [(w.word, w.flag) for w in words]
    except ImportError:
        print("jieba not installed, skipping POS analysis")
        return None


POS_CATEGORIES = {
    'noun': {'n', 'nr', 'ns', 'nt', 'nz', 'ng'},      # 名词
    'verb': {'v', 'vd', 'vn', 'vg'},                    # 动词
    'adj': {'a', 'ad', 'an', 'ag'},                      # 形容词
    'func': {'u', 'p', 'c', 'y', 'e', 'o', 'h', 'k'},  # 功能词（助词/介词/连词等）
    'punct': {'x', 'w'},                                  # 标点
    'num': {'m', 'q', 'mq'},                             # 数量词
}


def classify_pos(flag):
    for cat, flags in POS_CATEGORIES.items():
        if flag in flags:
            return cat
    return 'other'


# ============================================================
# 3. Token-级别分析
# ============================================================

def analyze_text(model, tokenizer, collector, text, device='cpu'):
    """对单段文本做完整分析，返回每个 token 的 E[K] + 语言学特征。"""

    # Tokenize
    input_ids = tokenizer(f"{tokenizer.bos_token}{text}", return_tensors='pt')['input_ids'].to(device)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
    seq_len = input_ids.shape[1]

    # Forward（收集 E[K]）
    collector.reset()
    for layer in model.layers:
        functional.reset_net(layer)
    functional.reset_net(model.output_neuron)

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits  # (1, seq_len, vocab)

    # E[K] per token
    ek_avg = collector.get_ek_per_token()  # (seq_len,)
    ek_per_layer = collector.get_ek_per_layer()

    # Surprisal: -log P(true token)
    log_probs = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)
    surprisals = []
    for i in range(seq_len - 1):
        next_token = input_ids[0, i + 1].item()
        surprisals.append(-log_probs[i, next_token].item())
    surprisals.append(0.0)  # 最后一个 token 没有 next

    # 词频（在 tokenizer vocab 中的 rank 近似）
    token_ids = input_ids[0].cpu().numpy()

    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'ek_avg': ek_avg,
        'ek_per_layer': ek_per_layer,
        'surprisals': np.array(surprisals),
        'seq_len': seq_len,
    }


# ============================================================
# 4. 批量分析 + 统计
# ============================================================

def run_analysis(model, tokenizer, collector, device='cpu'):
    """跑完整分析。"""

    test_texts = [
        "中国的首都是北京，位于华北平原的北部。",
        "人工智能是计算机科学的一个重要分支。",
        "今天天气很好，我们去公园散步吧。",
        "深度学习在自然语言处理领域取得了显著进展。",
        "数学是科学的基础，也是工程技术的重要工具。",
        "春天来了，花儿开了，小鸟在树上唱歌。",
        "量子计算可能在未来改变密码学的格局。",
        "他喜欢读书，尤其是历史和哲学方面的著作。",
    ]

    all_results = []
    pos_ek = defaultdict(list)  # {pos_category: [ek values]}
    all_surprisals = []
    all_eks = []

    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text}")

        result = analyze_text(model, tokenizer, collector, text, device)
        all_results.append(result)

        # Token-level 打印
        print(f"{'Token':<6} {'E[K]':>6} {'Surprisal':>10}")
        print("-" * 30)
        for i, (tok, ek, surp) in enumerate(zip(
            result['tokens'], result['ek_avg'], result['surprisals']
        )):
            print(f"{tok:<6} {ek:>6.2f} {surp:>10.2f}")

        all_surprisals.extend(result['surprisals'][:-1])  # 去掉最后一个
        all_eks.extend(result['ek_avg'][:-1])

        # POS 分析
        pos_tags = get_pos_tags(text)
        if pos_tags:
            # 将 jieba 分词结果映射到 tokenizer 的 token
            char_pos = []
            for word, flag in pos_tags:
                cat = classify_pos(flag)
                for ch in word:
                    char_pos.append((ch, cat))

            # 简单映射：每个 token 取其第一个字符的词性
            for i, tok in enumerate(result['tokens']):
                tok_clean = tok.strip()
                if not tok_clean or i >= len(result['ek_avg']):
                    continue
                # 在 char_pos 中找匹配
                for ch, cat in char_pos:
                    if ch in tok_clean:
                        pos_ek[cat].append(result['ek_avg'][i])
                        break

    # ============================================================
    # 5. 统计汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # POS vs E[K]
    print("\n--- E[K] by Part-of-Speech ---")
    print(f"{'Category':<12} {'Count':>6} {'Mean E[K]':>10} {'Std':>8}")
    for cat in ['noun', 'verb', 'adj', 'func', 'punct', 'num', 'other']:
        vals = pos_ek.get(cat, [])
        if vals:
            print(f"{cat:<12} {len(vals):>6} {np.mean(vals):>10.3f} {np.std(vals):>8.3f}")

    # Surprisal vs E[K] 相关性
    all_surprisals = np.array(all_surprisals)
    all_eks = np.array(all_eks)
    if len(all_surprisals) > 10:
        corr = np.corrcoef(all_surprisals, all_eks)[0, 1]
        print(f"\n--- Surprisal vs E[K] Correlation ---")
        print(f"Pearson r = {corr:.4f}  (n={len(all_surprisals)})")
        print(f"Mean surprisal: {all_surprisals.mean():.3f}")
        print(f"Mean E[K]: {all_eks.mean():.3f}")

    # 逐层 E[K] 分析
    print(f"\n--- Per-Layer E[K] (averaged across all texts) ---")
    layer_ek_block = []
    layer_ek_ffn = []
    for result in all_results:
        for layer_idx in sorted(result['ek_per_layer'].keys()):
            while len(layer_ek_block) <= layer_idx:
                layer_ek_block.append([])
                layer_ek_ffn.append([])
            if 'block' in result['ek_per_layer'][layer_idx]:
                layer_ek_block[layer_idx].append(np.mean(result['ek_per_layer'][layer_idx]['block']))
            if 'ffn' in result['ek_per_layer'][layer_idx]:
                layer_ek_ffn[layer_idx].append(np.mean(result['ek_per_layer'][layer_idx]['ffn']))

    print(f"{'Layer':>6} {'Block E[K]':>12} {'FFN E[K]':>12}")
    for i in range(len(layer_ek_block)):
        b = np.mean(layer_ek_block[i]) if layer_ek_block[i] else 0
        f = np.mean(layer_ek_ffn[i]) if layer_ek_ffn[i] else 0
        print(f"{i:>6} {b:>12.3f} {f:>12.3f}")

    # β 分布
    print(f"\n--- β Distribution (trained values) ---")
    for i, layer in enumerate(model.layers):
        beta_in1 = torch.sigmoid(layer.input_neuron1.w).detach().cpu().numpy()
        beta_in2 = torch.sigmoid(layer.input_neuron2.w).detach().cpu().numpy()
        b_beta = torch.sigmoid(layer.snn_block.b_beta.data).detach().cpu().numpy()
        print(f"  Layer {i:>2}: input1 β=[{beta_in1.min():.3f}, {beta_in1.max():.3f}] "
              f"input2 β=[{beta_in2.min():.3f}, {beta_in2.max():.3f}] "
              f"hidden β=[{b_beta.min():.3f}, {b_beta.max():.3f}]")

    # Firing rate 分析（通过 β 估算）
    print(f"\n--- Estimated Firing Characteristics ---")
    all_hidden_beta = []
    for layer in model.layers:
        b = torch.sigmoid(layer.snn_block.b_beta.data).detach().cpu().numpy()
        all_hidden_beta.append(b)
    all_hidden_beta = np.concatenate(all_hidden_beta)
    fast = (all_hidden_beta < 0.9).sum()
    slow = (all_hidden_beta >= 0.9).sum()
    print(f"  Total hidden neurons: {len(all_hidden_beta)}")
    print(f"  Fast (β<0.9): {fast} ({fast/len(all_hidden_beta)*100:.1f}%)")
    print(f"  Slow (β≥0.9): {slow} ({slow/len(all_hidden_beta)*100:.1f}%)")
    print(f"  β mean: {all_hidden_beta.mean():.4f}, std: {all_hidden_beta.std():.4f}")

    return all_results, pos_ek, all_surprisals, all_eks


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints_sft/ckpt_step6500.pth')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_snn/')
    model, config = load_model(args.ckpt, args.device)
    collector = EKCollector(model)

    results, pos_ek, surprisals, eks = run_analysis(model, tokenizer, collector, args.device)

    print("\n\nAnalysis complete.")
