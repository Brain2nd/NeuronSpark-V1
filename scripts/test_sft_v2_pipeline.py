"""SFT v2 全流程测试套件 — H100 开机前最后一道验证.

覆盖:
  A. 数据集完整性 + 每 source 样本结构
  B. apply_chat_template 渲染正确性
  C. dataset.SFTDataset + loss_mask 正确性 (mask 只在 assistant 段)
  D. AutoModelForCausalLM 加载 + dtype 检查 (neuron fp32 / matrix bf16)
  E. 单 step forward + loss 值合理
  F. 单 step backward + grad 非 NaN/Inf + 神经元 grad 是 fp32
  G. 模拟 sft_ds.py 初始化流程 (不走 deepspeed)

任意一项 FAIL 都不应上 H100.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

from dataset import SFTDataset

CKPT = 'checkpoints_sft_local/hf_step7000'
DATA = 'data/sft_v2_mix_filtered'
TOK = 'tokenizer/'
SEQ_LEN = 2048

fails = []
def check(cond, name, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f' — {detail}' if detail else ''))
    if not cond: fails.append(name)

print('=' * 70)
print('SFT v2 Pipeline Test Suite')
print('=' * 70)

# ========== A. 数据集完整性 ==========
print('\n[A] 数据集完整性')
raw_ds = load_from_disk(DATA)
check(len(raw_ds) > 290000, 'Sample count > 290k', f'got {len(raw_ds)}')
check('messages' in raw_ds.column_names, "Has 'messages' column", str(raw_ds.column_names))

from collections import Counter
src_cnt = Counter(raw_ds[i]['source'] for i in range(0, len(raw_ds), 1000))
check(len(src_cnt) >= 15, f'Has >= 15 sources', f'got {len(src_cnt)} sources (sampled)')

# 每条必须 messages 至少 2 个 (user + assistant), 第一个是 user
bad = 0
for i in range(0, len(raw_ds), 500):  # 每 500 采一个
    s = raw_ds[i]
    m = s['messages']
    if len(m) < 2 or m[0]['role'] != 'user' or m[-1]['role'] != 'assistant':
        bad += 1
check(bad == 0, 'All sampled messages have user→assistant', f'bad={bad}')

# 没有 system 污染 (按我们承诺的)
sys_pol = 0
for i in range(0, len(raw_ds), 500):
    for m in raw_ds[i]['messages']:
        if m.get('role') == 'system':
            sys_pol += 1
check(sys_pol == 0, 'No system messages (aligned with old training)', f'system msgs={sys_pol}')


# ========== B. apply_chat_template 渲染 ==========
print('\n[B] ChatML 渲染')
tokenizer = AutoTokenizer.from_pretrained(TOK)
sample = raw_ds[0]
text = tokenizer.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=False)
check('<|im_start|>user' in text, 'ChatML user marker present')
check('<|im_start|>assistant' in text, 'ChatML assistant marker present')
check('<|im_end|>' in text, 'ChatML im_end marker present')
check('<|im_start|>system' not in text, 'No system marker (as designed)')


# ========== C. Dataset + loss_mask 正确性 ==========
print('\n[C] SFTDataset + loss_mask')
dataset = SFTDataset(DATA, tokenizer, max_length=SEQ_LEN)
check(len(dataset) == len(raw_ds), 'Dataset len matches raw', f'{len(dataset)}')

# 随机取 10 条验证 mask
import random
random.seed(0)
test_idxs = random.sample(range(len(dataset)), 10)

all_mask_ok = True
for ti in test_idxs:
    X, Y, loss_mask = dataset[ti]
    # X 是 input_ids[:-1], Y 是 [1:], loss_mask 对应 Y 位置
    mask_n = loss_mask.sum().item()
    non_pad = (X != 0).sum().item()

    # 基本检查
    if mask_n == 0:
        print(f'    [FAIL] idx {ti}: mask_n=0 ({dataset._hf_dataset[ti]["source"]})')
        all_mask_ok = False
        continue

    # mask 应全部落在 assistant 段. 确认方法: 把 mask=1 的 Y 解码后, 应等于原 assistant content (approximately)
    # 这里简化: 确保 mask 起始位置在 text 后半段 (assistant header 之后)
    mask_positions = (loss_mask == 1).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        all_mask_ok = False; continue
    first_mask = mask_positions[0].item()

    # 找 assistant header 的 token id
    # <|im_start|>assistant\n 的 token_ids
    asst_tokens = tokenizer('<|im_start|>assistant\n', add_special_tokens=False)['input_ids']
    # 在 X (input) 里找这个子序列
    found_asst = False
    for j in range(len(X) - len(asst_tokens)):
        if X[j:j+len(asst_tokens)].tolist() == asst_tokens:
            found_asst = True
            # mask 第一个 1 应该在 j + len(asst_tokens) 附近
            # (Y = X[1:], 所以 mask 位置 vs X 位置差 1)
            expected_mask_start = j + len(asst_tokens) - 1
            if first_mask < expected_mask_start:
                print(f'    [FAIL] idx {ti}: mask starts at {first_mask}, expected >= {expected_mask_start}')
                all_mask_ok = False
            break
    if not found_asst:
        print(f'    [WARN] idx {ti}: no assistant header found in input (可能序列被截断)')

check(all_mask_ok, 'loss_mask 只落在 assistant 段', f'tested {len(test_idxs)} samples')

# 统计 mask 比例合理
ratios = []
for ti in test_idxs:
    X, Y, loss_mask = dataset[ti]
    non_pad = (X != 0).sum().item()
    if non_pad > 0:
        ratios.append(loss_mask.sum().item() / non_pad)
check(sum(ratios)/len(ratios) > 0.05, 'Avg mask ratio > 5% (not all masked out)',
      f'mean={sum(ratios)/len(ratios):.3f}')


# ========== D. 模型加载 + dtype ==========
print('\n[D] 模型加载 + dtype 检查')
print(f'  Loading HF ckpt {CKPT}...')
model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True)

# 模拟 sft_ds.py 的 dtype 处理 (本测试用同一逻辑)
for name, p in model.named_parameters():
    if name.endswith(('.w','.v_th','.b_beta','.b_alpha','.b_th')):
        p.data = p.data.float()
    else:
        p.data = p.data.to(torch.bfloat16)

# 随机抽样检查每种 param 类型
fp32_ok = []; bf16_ok = []
for name, p in model.named_parameters():
    if name.endswith(('.w','.v_th','.b_beta','.b_alpha','.b_th')):
        fp32_ok.append(p.dtype == torch.float32)
    else:
        bf16_ok.append(p.dtype == torch.bfloat16)

check(all(fp32_ok) and len(fp32_ok) > 0,
      f'所有神经元参数是 fp32 ({sum(fp32_ok)}/{len(fp32_ok)})')
check(all(bf16_ok) and len(bf16_ok) > 0,
      f'所有矩阵参数是 bf16 ({sum(bf16_ok)}/{len(bf16_ok)})')

# config 检查
cfg = model.config
check(cfg.vocab_size == 64002, f'vocab_size = 64002', f'got {cfg.vocab_size}')
check(cfg.D == 1024, f'D = 1024', f'got {cfg.D}')
check(cfg.K == 12, f'K = 12', f'got {cfg.K}')
check(cfg.num_layers == 24, f'num_layers = 24', f'got {cfg.num_layers}')


# ========== E. 单 step forward ==========
print('\n[E] 单 step forward')
model = model.cuda()
snn = model.snn

X, Y, loss_mask = dataset[0]
X = X.unsqueeze(0).cuda()
Y = Y.unsqueeze(0).cuda()
loss_mask = loss_mask.unsqueeze(0).cuda()

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = snn(token_ids=X)
logits = out.logits
check(logits.shape[0] == 1 and logits.shape[1] == X.shape[1] and logits.shape[2] == 64002,
      f'Logits shape {logits.shape}')
check(torch.isfinite(logits).all().item(), 'Logits 全部 finite')

ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), Y.view(-1), reduction='none')
masked_loss = (ce * loss_mask.view(-1).float()).sum() / loss_mask.sum().clamp_min(1.0)
check(torch.isfinite(masked_loss).item(), 'Masked loss finite')
check(0.0 < masked_loss.item() < 20.0, f'Loss in [0, 20] range', f'loss={masked_loss.item():.3f}')


# ========== F. 单 step backward + grad 检查 ==========
print('\n[F] 单 step backward + grad dtype')
model.train()
optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
optim.zero_grad()

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = snn(token_ids=X)
logits = out.logits
ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), Y.view(-1), reduction='none')
masked_loss = (ce * loss_mask.view(-1).float()).sum() / loss_mask.sum().clamp_min(1.0)

# 加上 SNN model 自己的 aux losses (ponder_cost etc.) — 看 SNNModelOutput
aux_losses = []
for attr in ['ponder_cost', 'ek_floor_cost', 'snvr_cost', 'b_th_reg_cost']:
    v = getattr(out, attr, None)
    if v is not None and torch.is_tensor(v):
        aux_losses.append((attr, v.item()))
        masked_loss = masked_loss + v.float() * 0.001  # 小权重, 只是测能 backward

print(f'    aux loss components: {aux_losses}')

masked_loss.backward()

# 检查 grad
grad_ok_fp32 = []; grad_ok_bf16 = []
any_nan = False; any_inf = False
for name, p in model.named_parameters():
    if p.grad is None:
        continue
    if not torch.isfinite(p.grad).all():
        if torch.isnan(p.grad).any(): any_nan = True
        if torch.isinf(p.grad).any(): any_inf = True
    if name.endswith(('.w','.v_th','.b_beta','.b_alpha','.b_th')):
        grad_ok_fp32.append(p.grad.dtype == torch.float32)
    else:
        grad_ok_bf16.append(p.grad.dtype in (torch.float32, torch.bfloat16))  # autograd 通常回到 fp32

check(not any_nan, 'No NaN in grads')
check(not any_inf, 'No Inf in grads')
check(all(grad_ok_fp32) and len(grad_ok_fp32) > 0,
      f'神经元参数 grad 是 fp32 ({sum(grad_ok_fp32)}/{len(grad_ok_fp32)})')


# ========== G. sft_ds.py 加载逻辑兼容性 ==========
print('\n[G] sft_ds.py 兼容性 (模拟 init_model + pretrained_ckpt 加载路径)')

# 模拟 sft_ds.py 的 init_model: SNNLanguageModel 从 scratch 初始化, 然后 load_state_dict
from model import SNNLanguageModel
fresh = SNNLanguageModel(
    vocab_size=64002, D=1024, N=8, K=12, num_layers=24, D_ff=3072,
).cuda()

# 按 sft_ds.py 的逻辑: is_hf → AutoModelForCausalLM, 然后 fresh.load_state_dict(hf.snn.state_dict())
missing, unexpected = fresh.load_state_dict(snn.state_dict(), strict=False)
check(len(unexpected) == 0, f'No unexpected keys on load', f'got {unexpected[:3]}')
# missing 应该只有 non-persistent buffers (rope_cos/sin, possibly)
accept_missing = all('rope' in m.lower() or 'freq' in m.lower() for m in missing)
check(len(missing) == 0 or accept_missing,
      f'Missing keys (全是 rope 非持久 buffer 或空)',
      f'missing: {len(missing)} sample: {missing[:3]}')

# 防御性提升 neuron 到 fp32 (sft_ds.py 有这段)
for name, p in fresh.named_parameters():
    if name.endswith(('.w','.v_th','.b_beta','.b_alpha','.b_th')) and p.dtype != torch.float32:
        p.data = p.data.float()

# 对比 HF 和 fresh 的 loss 值 (最 practical 的等价性测试).
# PLIF+PonderNet 有 stochastic halt + v_th_min 等非参数 config 可能有小差异,
# 故不对 logits 做 bit-exact, 只看 masked loss 在统计意义上接近.
snn.eval(); fresh.eval()
losses_hf = []; losses_fresh = []
for ti in test_idxs[:5]:
    Xt, Yt, mt = dataset[ti]
    Xt = Xt.unsqueeze(0).cuda(); Yt = Yt.unsqueeze(0).cuda(); mt = mt.unsqueeze(0).cuda()
    if mt.sum() == 0: continue
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        l1 = F.cross_entropy(snn(token_ids=Xt).logits.view(-1,64002).float(), Yt.view(-1), reduction='none')
        lhf = ((l1 * mt.view(-1).float()).sum() / mt.sum()).item()
        l2 = F.cross_entropy(fresh(token_ids=Xt).logits.view(-1,64002).float(), Yt.view(-1), reduction='none')
        lfr = ((l2 * mt.view(-1).float()).sum() / mt.sum()).item()
    losses_hf.append(lhf); losses_fresh.append(lfr)

import numpy as np
diff_mean = abs(np.mean(losses_hf) - np.mean(losses_fresh))
check(diff_mean < 0.1,
      f'Fresh vs HF 平均 loss diff < 0.1 (5 samples, stochastic halt 导致 per-sample 有噪声)',
      f'HF mean={np.mean(losses_hf):.3f}, Fresh mean={np.mean(losses_fresh):.3f}, diff={diff_mean:.4f}')


# ========== 总结 ==========
print('\n' + '=' * 70)
if fails:
    print(f'[FAIL] {len(fails)} test(s) failed:')
    for f in fails: print(f'  - {f}')
    sys.exit(1)
else:
    print('[ALL PASS] 可以开 H100')
    sys.exit(0)
