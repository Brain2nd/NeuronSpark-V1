"""
Surprisal vs E[K] analysis with scatter plot + binned bar chart.
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional
from model import SNNLanguageModel

# Load
ckpt = torch.load('checkpoints_sft/ckpt_step6500.pth', map_location='cpu', weights_only=False)
config = ckpt.get('model_config', {})
model = SNNLanguageModel(**{k: config[k] for k in ['vocab_size','D','N','K','num_layers','D_ff']})
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('./tokenizer_snn/')
print(f"Model loaded: D={config['D']}, K={config['K']}, L={config['num_layers']}")

# Hook
class EKCollector:
    def __init__(self, model):
        self.ek_data = {}
        for idx, layer in enumerate(model.layers):
            orig = layer._adaptive_aggregate
            cc = [0]
            def make_hook(i, o, c):
                def hooked(frames, halt_proj):
                    agg, pc, ek = o(frames, halt_proj)
                    sub = 'block' if c[0]%2==0 else 'ffn'
                    if i not in self.ek_data: self.ek_data[i] = {}
                    self.ek_data[i][sub] = ek.cpu().numpy()
                    c[0] += 1
                    return agg, pc, ek
                return hooked
            layer._adaptive_aggregate = make_hook(idx, orig, cc)
    def reset(self): self.ek_data = {}
    def get_ek_per_token(self):
        all_ek = []
        for li in sorted(self.ek_data.keys()):
            for sub in ['block','ffn']:
                if sub in self.ek_data[li]:
                    all_ek.append(self.ek_data[li][sub][:,0])
        return np.mean(all_ek, axis=0) if all_ek else None

collector = EKCollector(model)

texts = [
    "中国的首都是北京，位于华北平原的北部。",
    "人工智能是计算机科学的一个重要分支。",
    "深度学习在自然语言处理领域取得了显著进展。",
    "量子计算可能在未来改变密码学的格局。",
    "机器学习算法可以从大量数据中自动发现规律。",
    "神经网络的训练需要大量的计算资源和数据。",
    "自动驾驶技术正在快速发展，但安全问题仍需解决。",
    "云计算为企业提供了灵活的基础设施服务。",
    "大数据分析帮助企业做出更好的商业决策。",
    "物联网将各种设备连接到互联网，实现智能控制。",
    "今天天气很好，我们去公园散步吧。",
    "春天来了，花儿开了，小鸟在树上唱歌。",
    "他喜欢读书，尤其是历史和哲学方面的著作。",
    "这家餐厅的菜做得非常好吃，价格也很合理。",
    "周末我们一家人去郊外野餐，孩子们玩得很开心。",
    "早上起来先喝一杯温水，对身体有好处。",
    "下雨了，记得带伞出门，注意安全。",
    "图书馆里很安静，适合读书和学习。",
    "数学是科学的基础，也是工程技术的重要工具。",
    "地球围绕太阳运转，一年大约需要三百六十五天。",
    "水在零度以下会结冰，在一百度以上会沸腾。",
    "光速是宇宙中最快的速度，约为每秒三十万公里。",
    "历史的发展是一个不断前进的过程。",
    "教育是国家发展的根本，人才是最重要的资源。",
    "读万卷书不如行万里路，实践出真知。",
    "经济全球化促进了国际贸易的快速增长。",
    "城市化进程加快，越来越多的人涌入大城市。",
    "环境保护是全人类共同的责任和义务。",
    "医疗技术的进步延长了人类的平均寿命。",
    "互联网改变了人们的生活方式和工作方式。",
    "文化交流有助于增进不同国家之间的相互理解。",
    "可持续发展是当今世界面临的重要课题。",
    "虽然这个项目面临很多困难，但是团队成员都非常努力，最终取得了成功。",
    "随着科技的不断发展，人们的生活水平得到了显著提高，但同时也带来了一些新的问题。",
    "在过去的几十年里，中国经济取得了举世瞩目的成就，成为世界第二大经济体。",
    "人工智能不仅可以帮助我们解决复杂的问题，还可以提高工作效率和生活质量。",
    "教育改革的目标是培养具有创新精神和实践能力的高素质人才。",
    "全球气候变化已经成为影响人类生存和发展的重大挑战。",
    "科学研究需要严谨的态度和坚持不懈的精神。",
    "互联网的普及使得信息传播的速度和范围大大增加。",
]

# Collect
all_surp_bos, all_ek_bos = [], []
all_surp_no_bos, all_ek_no_bos = [], []
is_bos = []

for text in texts:
    collector.reset()
    input_ids = tokenizer(f"{tokenizer.bos_token}{text}", return_tensors='pt')['input_ids']
    for layer in model.layers: functional.reset_net(layer)
    functional.reset_net(model.output_neuron)
    with torch.no_grad():
        out = model(input_ids)
    ek_avg = collector.get_ek_per_token()
    log_probs = F.log_softmax(out.logits[0], dim=-1)
    seq_len = input_ids.shape[1]
    for i in range(seq_len - 1):
        surp = -log_probs[i, input_ids[0, i+1].item()].item()
        ek = ek_avg[i]
        all_surp_bos.append(surp)
        all_ek_bos.append(ek)
        is_bos.append(i == 0)
        if i > 0:
            all_surp_no_bos.append(surp)
            all_ek_no_bos.append(ek)

all_surp_bos = np.array(all_surp_bos)
all_ek_bos = np.array(all_ek_bos)
all_surp_no_bos = np.array(all_surp_no_bos)
all_ek_no_bos = np.array(all_ek_no_bos)
is_bos = np.array(is_bos)

r_all = np.corrcoef(all_surp_bos, all_ek_bos)[0,1]
r_no_bos = np.corrcoef(all_surp_no_bos, all_ek_no_bos)[0,1]
print(f"r (with BOS) = {r_all:.4f}, r (without BOS) = {r_no_bos:.4f}")
print(f"n = {len(all_surp_bos)} (with BOS), {len(all_surp_no_bos)} (without)")

# ============================================================
# Plot: 2-panel figure
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# (a) Scatter: surprisal vs E[K], BOS highlighted
non_bos_mask = ~is_bos
ax1.scatter(all_surp_bos[non_bos_mask], all_ek_bos[non_bos_mask],
            alpha=0.3, s=15, color='#3498db', label=f'Content tokens (n={non_bos_mask.sum()})')
ax1.scatter(all_surp_bos[is_bos], all_ek_bos[is_bos],
            alpha=0.9, s=60, color='#e74c3c', marker='X', zorder=10,
            label=f'BOS tokens (n={is_bos.sum()})')

# Trend lines
z_all = np.polyfit(all_surp_bos, all_ek_bos, 1)
x_line = np.linspace(0, all_surp_bos.max(), 100)
ax1.plot(x_line, np.polyval(z_all, x_line), 'r--', alpha=0.5, linewidth=1.5,
         label=f'All tokens: r={r_all:.2f}')

z_no = np.polyfit(all_surp_no_bos, all_ek_no_bos, 1)
x_line2 = np.linspace(0, all_surp_no_bos.max(), 100)
ax1.plot(x_line2, np.polyval(z_no, x_line2), 'b-', alpha=0.7, linewidth=1.5,
         label=f'Excl. BOS: r={r_no_bos:.2f}')

ax1.set_xlabel('Surprisal (-log P(next token))', fontsize=12)
ax1.set_ylabel('E[K] (expected SNN steps)', fontsize=12)
ax1.set_title('(a) Surprisal vs E[K]: Scatter', fontsize=13)
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.annotate('BOS: low E[K], high surprisal\n(dominates negative correlation)',
             xy=(8.5, 3.5), fontsize=8, color='#e74c3c', fontstyle='italic',
             ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce4ec', alpha=0.8))

# (b) Binned bar chart (excluding BOS)
bins = [(0,1,'[0,1)'), (1,2,'[1,2)'), (2,3,'[2,3)'), (3,5,'[3,5)'), (5,7,'[5,7)'), (7,12,'[7,12)')]
bin_means = []
bin_stds = []
bin_labels = []
bin_counts = []
for lo, hi, label in bins:
    mask = (all_surp_no_bos >= lo) & (all_surp_no_bos < hi)
    if mask.sum() >= 2:
        bin_means.append(all_ek_no_bos[mask].mean())
        bin_stds.append(all_ek_no_bos[mask].std())
        bin_labels.append(label)
        bin_counts.append(mask.sum())

x_pos = np.arange(len(bin_labels))
bars = ax2.bar(x_pos, bin_means, yerr=bin_stds, capsize=4,
               color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=0.5)
for i, (m, c) in enumerate(zip(bin_means, bin_counts)):
    ax2.text(i, m + bin_stds[i] + 0.05, f'n={c}', ha='center', fontsize=8, color='gray')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(bin_labels)
ax2.set_xlabel('Surprisal Range', fontsize=12)
ax2.set_ylabel('Mean E[K] (expected SNN steps)', fontsize=12)
ax2.set_title('(b) E[K] by Surprisal Bin (excl. BOS)', fontsize=13)
ax2.set_ylim(6.5, 8.5)
ax2.axhline(y=all_ek_no_bos.mean(), color='gray', ls='--', alpha=0.5,
            label=f'overall mean={all_ek_no_bos.mean():.2f}')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/surprisal_vs_ek.pdf', dpi=150, bbox_inches='tight')
plt.savefig('paper/figures/surprisal_vs_ek.png', dpi=150, bbox_inches='tight')
print('Saved paper/figures/surprisal_vs_ek.pdf/png')
