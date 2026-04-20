"""本地 dry-run: 加载 sft_v2_mix_filtered + hf_step7000, 跑 10 step,
检查 loss_mask / pad / eos / loss 值合理性. 不用 DeepSpeed, 直接 PyTorch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from dataset import SFTDataset

CKPT = 'checkpoints_sft_local/hf_step7000'
DATA = 'data/sft_v2_mix_filtered'
TOK  = 'tokenizer/'
SEQ_LEN = 2048
BS = 1
STEPS = 10

print(f'Loading tokenizer {TOK}...')
tokenizer = AutoTokenizer.from_pretrained(TOK)

print(f'Loading model {CKPT}...')
model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True)
# neuron fp32, matrices bf16
for name, p in model.named_parameters():
    if name.endswith(('.w','.v_th','.b_beta','.b_alpha','.b_th')):
        p.data = p.data.float()
    else:
        p.data = p.data.to(torch.bfloat16)
model = model.cuda().eval()  # eval for dry-run, only check forward + loss
snn = model.snn

print(f'Loading dataset {DATA}...')
dataset = SFTDataset(DATA, tokenizer, max_length=SEQ_LEN)
print(f'  n={len(dataset)}')

# 取前 STEPS 个 sample, 手动跑 forward + 算 loss
print(f'\n=== 运行 {STEPS} 个 sample 的 forward + CE loss ===')
print(f'{"i":<3} {"src":<22} {"tok_len":<8} {"mask_n":<8} {"loss":<8} {"notes"}')

from collections import Counter
src_cnt = Counter()
issues = []

for i in range(STEPS):
    sample = dataset._hf_dataset[i]
    src = sample['source']
    src_cnt[src] += 1

    # 手动用 dataset 的 tokenize + mask 流程
    X, Y, loss_mask = dataset[i]
    X = X.unsqueeze(0).cuda()
    Y = Y.unsqueeze(0).cuda()
    loss_mask = loss_mask.unsqueeze(0).cuda()

    # 有效 token 数 (mask=1)
    mask_n = loss_mask.sum().item()
    tok_len = (X != 0).sum().item()  # non-pad 数

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            out = snn(token_ids=X)
    logits = out.logits  # (1, seq_len, vocab)
    # CE loss
    shift_logits = logits.view(-1, logits.size(-1))
    shift_labels = Y.view(-1)
    shift_mask = loss_mask.view(-1)
    ce = torch.nn.functional.cross_entropy(
        shift_logits.float(), shift_labels, reduction='none',
    )
    masked_loss = (ce * shift_mask.float()).sum() / shift_mask.sum().clamp_min(1.0)
    loss = masked_loss.item()

    notes = []
    if mask_n == 0:
        notes.append('MASK=0!')
    if loss > 10:
        notes.append(f'LOSS高!')
    if tok_len < 10:
        notes.append('tok少')
    if mask_n > 0 and loss > 20:
        issues.append(f'i={i} src={src} loss={loss:.2f}')

    notes_str = ','.join(notes) if notes else 'ok'
    print(f'{i:<3} {src:<22} {tok_len:<8} {mask_n:<8} {loss:<8.3f} {notes_str}')

print(f'\nSource 分布 (前 {STEPS}): {dict(src_cnt)}')
if issues:
    print(f'\nISSUES: {issues}')
else:
    print('\n[PASS] 无异常')

# 统计各类 loss 均值 (多采 50 条)
print(f'\n=== 额外 50 sample 按 source 统计 mean loss ===')
src_loss = {}
for i in range(50):
    s = dataset._hf_dataset[i]['source']
    X, Y, loss_mask = dataset[i]
    X = X.unsqueeze(0).cuda(); Y = Y.unsqueeze(0).cuda(); loss_mask = loss_mask.unsqueeze(0).cuda()
    if loss_mask.sum() == 0:
        continue
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            out = snn(token_ids=X)
    logits = out.logits
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), Y.view(-1), reduction='none')
    ml = (ce * loss_mask.view(-1).float()).sum() / loss_mask.sum().clamp_min(1.0)
    src_loss.setdefault(s, []).append(ml.item())

for s, ls in src_loss.items():
    import numpy as np
    a = np.array(ls)
    print(f'  {s:<25} n={len(a)} mean={a.mean():.3f} std={a.std():.3f}')

print('\n[DONE]')
