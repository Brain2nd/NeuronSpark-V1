from tbparse import SummaryReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re, os

# 1. V1 text log
steps_v1, losses_v1 = [], []
with open('train_v80.log') as f:
    for line in f:
        m = re.match(r'Epoch:\[1/1\]\((\d+)/\d+\) loss:([\d.]+)', line)
        if m:
            steps_v1.append(int(m.group(1)))
            losses_v1.append(float(m.group(2)))
steps_v1 = np.array(steps_v1)
losses_v1 = np.array(losses_v1)

# 2. Ablation TB runs
runs_dir = '/tmp/tb_runs/runs'
skip = {'fsdp_1faa7df'}  # V2 main training, not ablation

# Map commit hash to readable name
labels = {
    'fsdp_68d58fe': 'MPD-AGL + no Phase 2',
    'fsdp_f730d52': 'E[K] floor',
    'fsdp_276b9a8': 'Bounded α gain',
    'fsdp_2f57f7b': 'HC α (decoupled)',
    'fsdp_b2c1686': 'Sinkhorn health',
    'fsdp_4af7a79': 'Cortical lateral',
    'fsdp_d427b32': 'Baseline debug (NaN)',
    'fsdp_bfe8122': 'Param alignment',
    'fsdp_abfb023': 'Dashboard merge',
}

ablation_runs = {}
for d in sorted(os.listdir(runs_dir)):
    if d in skip: continue
    run_path = os.path.join(runs_dir, d)
    if not os.path.isdir(run_path): continue
    try:
        reader = SummaryReader(run_path)
        df = reader.scalars
        loss_df = df[df['tag']=='train/loss'].sort_values('step')
        v = loss_df['value'].values
        s = loss_df['step'].values
        f = np.isfinite(v)
        has_nan = f.sum() < len(v)
        if f.sum() > 0:
            ablation_runs[d] = (s[f], v[f], has_nan)
    except: pass

# 3. Two-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# === Left: Full view (0-85K) showing V1 dominance ===
colors_ablation = {
    'fsdp_68d58fe': '#e74c3c',
    'fsdp_f730d52': '#e67e22',
    'fsdp_276b9a8': '#f39c12',
    'fsdp_2f57f7b': '#9b59b6',
    'fsdp_b2c1686': '#1abc9c',
    'fsdp_4af7a79': '#3498db',
    'fsdp_d427b32': '#95a5a6',
    'fsdp_bfe8122': '#bdc3c7',
    'fsdp_abfb023': '#7f8c8d',
}

for name, (s, v, has_nan) in ablation_runs.items():
    c = colors_ablation.get(name, 'gray')
    ax1.plot(s/1000, v, color=c, alpha=0.6, linewidth=1.2)

w = 20
v1s = np.convolve(losses_v1, np.ones(w)/w, mode='valid')
s1s = steps_v1[w-1:]
ax1.plot(s1s/1000, v1s, color='blue', linewidth=2.5, zorder=10, label='V1 final (loss → 3.5)')
ax1.set_xlabel('Training Steps (K)', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('(a) Full Training View', fontsize=13)
ax1.set_ylim(2.5, 9.5)
ax1.axhline(y=7.0, color='gray', ls=':', alpha=0.4)
ax1.text(50, 7.1, 'loss = 7.0 (no ablation breaks this)', fontsize=8, color='gray')
ax1.legend(fontsize=10, loc='right')
ax1.grid(True, alpha=0.3)

# === Right: Zoomed view (0-13K) showing ablation detail ===
for name, (s, v, has_nan) in ablation_runs.items():
    c = colors_ablation.get(name, 'gray')
    label = labels.get(name, name)
    if has_nan:
        label += ' (NaN)'
    ax2.plot(s/1000, v, color=c, alpha=0.8, linewidth=1.5, label=label)

# V1 in same range
mask = steps_v1 <= 13000
ax2.plot(steps_v1[mask]/1000, losses_v1[mask], color='blue', linewidth=2.5,
         alpha=0.8, zorder=10, label='V1 final')

ax2.set_xlabel('Training Steps (K)', fontsize=12)
ax2.set_ylabel('Training Loss', fontsize=12)
ax2.set_title('(b) Zoomed: Ablation Region (0–13K steps)', fontsize=13)
ax2.set_xlim(0, 13)
ax2.set_ylim(6.0, 9.5)
ax2.axhline(y=7.5, color='red', ls='--', alpha=0.3)
ax2.text(8, 7.55, 'divergence', fontsize=8, color='red', alpha=0.5)
ax2.legend(fontsize=7.5, loc='lower left', ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/ablation_runs.pdf', dpi=150, bbox_inches='tight')
plt.savefig('paper/figures/ablation_runs.png', dpi=150, bbox_inches='tight')
print('Done')
