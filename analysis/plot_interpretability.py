"""
生物可解释性分析图：PonderNet E[K] vs 语言学特征
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional
from model import SNNLanguageModel
import jieba.posseg as pseg

# ============================================================
# Setup
# ============================================================

# CJK font for Chinese token labels
import matplotlib.font_manager as fm
cjk_fonts = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ['Heiti', 'PingFang', 'Songti', 'STSong', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK', 'WenQuanYi'])]
cjk_font = cjk_fonts[0] if cjk_fonts else None
print(f"CJK font: {cjk_font}")

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})
if cjk_font:
    plt.rcParams['font.sans-serif'] = [cjk_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False

# Load model
ckpt = torch.load('checkpoints_sft/ckpt_step6500.pth', map_location='cpu', weights_only=False)
config = ckpt.get('model_config', {})
model = SNNLanguageModel(**{k: config[k] for k in ['vocab_size','D','N','K','num_layers','D_ff']})
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('./tokenizer_snn/')
print(f"Model loaded: D={config['D']}, K={config['K']}, L={config['num_layers']}")

# Hook to collect E[K]
class EKCollector:
    def __init__(self, model):
        self.model = model
        self.ek_data = {}
        for idx, layer in enumerate(model.layers):
            orig = layer._adaptive_aggregate
            call_count = [0]
            def make_hook(i, o, cc):
                def hooked(frames, halt_proj):
                    agg, pc, ek = o(frames, halt_proj)
                    sub = 'block' if cc[0] % 2 == 0 else 'ffn'
                    if i not in self.ek_data: self.ek_data[i] = {}
                    self.ek_data[i][sub] = ek.cpu().numpy()
                    cc[0] += 1
                    return agg, pc, ek
                return hooked
            layer._adaptive_aggregate = make_hook(idx, orig, call_count)

    def reset(self): self.ek_data = {}

    def get_ek_per_token(self):
        all_ek = []
        for li in sorted(self.ek_data.keys()):
            for sub in ['block', 'ffn']:
                if sub in self.ek_data[li]:
                    all_ek.append(self.ek_data[li][sub][:, 0])
        return np.mean(all_ek, axis=0) if all_ek else None

    def get_ek_per_layer(self):
        result = {}
        for li in sorted(self.ek_data.keys()):
            result[li] = {}
            for sub in ['block', 'ffn']:
                if sub in self.ek_data[li]:
                    result[li][sub] = self.ek_data[li][sub][:, 0]
        return result

collector = EKCollector(model)

# POS categories
POS_CATEGORIES = {
    'Noun': {'n','nr','ns','nt','nz','ng'},
    'Verb': {'v','vd','vn','vg'},
    'Adj': {'a','ad','an','ag'},
    'Function': {'u','p','c','y','e','o','h','k'},
    'Punct': {'x','w'},
    'Number': {'m','q','mq'},
}

def classify_pos(flag):
    for cat, flags in POS_CATEGORIES.items():
        if flag in flags: return cat
    return 'Other'

# ============================================================
# Collect data
# ============================================================

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

pos_ek = defaultdict(list)
all_layer_ek_block = defaultdict(list)
all_layer_ek_ffn = defaultdict(list)
example_tokens = []
example_ek = []
example_text_idx = 0  # use first text as example

for ti, text in enumerate(test_texts):
    collector.reset()
    input_ids = tokenizer(f"{tokenizer.bos_token}{text}", return_tensors='pt')['input_ids']
    for layer in model.layers: functional.reset_net(layer)
    functional.reset_net(model.output_neuron)
    with torch.no_grad():
        out = model(input_ids)

    ek_avg = collector.get_ek_per_token()
    ek_per_layer = collector.get_ek_per_layer()
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    if ti == example_text_idx:
        example_tokens = tokens[1:]  # skip BOS
        example_ek = ek_avg[1:]

    # Per-layer
    for li in sorted(ek_per_layer.keys()):
        if 'block' in ek_per_layer[li]:
            all_layer_ek_block[li].append(np.mean(ek_per_layer[li]['block']))
        if 'ffn' in ek_per_layer[li]:
            all_layer_ek_ffn[li].append(np.mean(ek_per_layer[li]['ffn']))

    # POS mapping
    pos_tags = list(pseg.cut(text))
    char_pos = []
    for word, flag in pos_tags:
        cat = classify_pos(flag)
        for ch in word:
            char_pos.append((ch, cat))

    for i, tok in enumerate(tokens):
        if i >= len(ek_avg): continue
        tok_clean = tok.strip()
        if not tok_clean: continue
        for ch, cat in char_pos:
            if ch in tok_clean:
                pos_ek[cat].append(ek_avg[i])
                break

print(f"Collected data from {len(test_texts)} texts")

# β data
all_hidden_beta = []
layer_beta_stats = []
for layer in model.layers:
    b = torch.sigmoid(layer.snn_block.b_beta.data).detach().cpu().numpy()
    all_hidden_beta.append(b)
    layer_beta_stats.append((b.mean(), b.std(), b.min(), b.max()))
all_hidden_beta = np.concatenate(all_hidden_beta)

# ============================================================
# Plot: 4-panel figure
# ============================================================

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ---- (a) E[K] per token example ----
ax1 = fig.add_subplot(gs[0, 0])
colors_tok = []
for tok in example_tokens:
    # color by rough category
    tok_clean = tok.strip()
    is_punct = all(not c.isalnum() for c in tok_clean) if tok_clean else False
    if is_punct:
        colors_tok.append('#e74c3c')
    else:
        colors_tok.append('#3498db')

bars = ax1.bar(range(len(example_ek)), example_ek, color=colors_tok, alpha=0.8, edgecolor='white', linewidth=0.5)
ax1.set_xticks(range(len(example_tokens)))
ax1.set_xticklabels(example_tokens, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('E[K] (expected SNN steps)')
ax1.set_title('(a) Per-Token E[K]: "' + test_texts[0][:12] + '..."')
ax1.axhline(y=np.mean(example_ek), color='gray', ls='--', alpha=0.5, label=f'mean={np.mean(example_ek):.1f}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# ---- (b) E[K] by POS category ----
ax2 = fig.add_subplot(gs[0, 1])
cats_order = ['Noun', 'Verb', 'Adj', 'Function', 'Punct', 'Number', 'Other']
cats_present = [c for c in cats_order if c in pos_ek and len(pos_ek[c]) >= 2]
means = [np.mean(pos_ek[c]) for c in cats_present]
stds = [np.std(pos_ek[c]) for c in cats_present]
counts = [len(pos_ek[c]) for c in cats_present]

bar_colors = ['#2ecc71', '#3498db', '#e67e22', '#95a5a6', '#e74c3c', '#9b59b6', '#bdc3c7']
bars2 = ax2.bar(range(len(cats_present)), means, yerr=stds, capsize=4,
                color=bar_colors[:len(cats_present)], alpha=0.85, edgecolor='white', linewidth=0.5)

# annotate counts
for i, (m, c) in enumerate(zip(means, counts)):
    ax2.text(i, m + stds[i] + 0.1, f'n={c}', ha='center', fontsize=8, color='gray')

ax2.set_xticks(range(len(cats_present)))
ax2.set_xticklabels(cats_present)
ax2.set_ylabel('E[K] (expected SNN steps)')
ax2.set_title('(b) E[K] by Part-of-Speech Category')
ax2.grid(axis='y', alpha=0.3)

# ---- (c) Per-layer E[K] heatmap-style ----
ax3 = fig.add_subplot(gs[1, 0])
num_layers = len(all_layer_ek_block)
layers = list(range(num_layers))
block_means = [np.mean(all_layer_ek_block[i]) for i in layers]
ffn_means = [np.mean(all_layer_ek_ffn[i]) for i in layers]

x = np.arange(num_layers)
width = 0.35
bars_b = ax3.bar(x - width/2, block_means, width, label='SNNBlock', color='#3498db', alpha=0.85)
bars_f = ax3.bar(x + width/2, ffn_means, width, label='SNNFFN', color='#e67e22', alpha=0.85)

ax3.set_xlabel('Layer Index')
ax3.set_ylabel('E[K] (expected SNN steps)')
ax3.set_title('(c) Per-Layer E[K]: Block vs FFN')
ax3.set_xticks(x[::2])
ax3.set_xticklabels([str(i) for i in layers[::2]])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# ---- (d) β distribution histogram ----
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(all_hidden_beta, bins=50, color='#2ecc71', alpha=0.8, edgecolor='white', linewidth=0.5, density=True)
ax4.axvline(x=0.9, color='red', ls='--', linewidth=1.5, label=f'β=0.9 threshold')
fast_pct = (all_hidden_beta < 0.9).mean() * 100
slow_pct = (all_hidden_beta >= 0.9).mean() * 100
ax4.text(0.55, ax4.get_ylim()[1]*0.85, f'Fast (β<0.9): {fast_pct:.1f}%', fontsize=10, color='#2ecc71', fontweight='bold')
ax4.text(0.91, ax4.get_ylim()[1]*0.85, f'Slow: {slow_pct:.1f}%', fontsize=10, color='red', fontweight='bold')
ax4.set_xlabel('β (membrane decay rate)')
ax4.set_ylabel('Density')
ax4.set_title(f'(d) Hidden Neuron β Distribution (n={len(all_hidden_beta):,})')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.savefig('paper/figures/interpretability.pdf', bbox_inches='tight')
plt.savefig('paper/figures/interpretability.png', bbox_inches='tight')
print('Saved paper/figures/interpretability.pdf/png')
