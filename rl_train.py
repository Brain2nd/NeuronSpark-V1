"""
SNN 语言模型 RL 训练（REINFORCE + 关键词奖励）

对话式 RL：模型自由回答知识性问题，用关键词匹配判断回复是否包含正确信息。

数据格式（由 scripts/build_rl_domain_data.py 构建）：
  prompt_messages: [{"role":"system",...}, {"role":"user","content":"水的化学式是什么？"}]
  keywords: ["H2O", "H₂O"]
  domain: "化学"

奖励设计：
  R = R_keyword + R_quality
  - R_keyword:  包含任一关键词 → +1.0, 不包含 → -0.5
  - R_quality:  重复惩罚(-0.5~0), 长度惩罚, 空输出(-1.0)

用法：
    python rl_train.py \
        --sft_ckpt checkpoints_sft/ckpt_step11000 \
        --data_path data/rl-domain \
        --vocab_size 64002 --D 1024 --N 8 --K 12 --num_layers 24 --D_ff 3072
"""

import os
import re
import math
import time
import json
import argparse
import warnings
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict

from model import SNNLanguageModel
from checkpoint_utils import save_checkpoint, load_model_weights
from atomic_ops.snn_base import functional

warnings.filterwarnings('ignore')


# ============================================================
# 奖励函数（对话式关键词匹配）
# ============================================================

class RewardFunction:
    """对话式 RL 奖励函数。

    总奖励 R = R_keyword + R_quality

    R_keyword (核心, 范围 [-0.5, +1.0]):
      - 回复中包含任一正确关键词 → +1.0
      - 包含多个关键词 → +1.0 + 0.2 * (额外命中数)，上限 +1.5
      - 不包含任何关键词 → -0.5

    R_quality (输出质量, 范围 [-1.0, +0.2]):
      - 空输出 → -1.0
      - 重复退化（4-gram 重复率 > 70%）→ -0.5
      - 过短 (<5 字符) → -0.3
      - 过长 (>500 字符) → -0.1
      - 合理长度 (10-200 字符) → +0.1
      - 包含问题本身的关键信息（相关性）→ +0.1
    """

    def compute(self, generated_text, keywords, question=""):
        """计算总奖励。

        Args:
            generated_text: 模型生成的回复
            keywords: list[str], 正确答案关键词列表（包含任一即可）
            question: 原始问题文本（用于相关性检查）

        Returns:
            total_reward: float
            detail: dict 奖励分解
        """
        detail = {'r_keyword': 0.0, 'r_quality': 0.0}

        # 空输出
        if not generated_text or not generated_text.strip():
            detail['r_quality'] = -1.0
            detail['r_keyword'] = -0.5
            return sum(detail.values()), detail

        text = generated_text.strip()

        # ---- R_keyword: 关键词匹配 ----
        text_lower = text.lower()
        hits = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                hits += 1

        if hits == 0:
            detail['r_keyword'] = -0.5
        elif hits == 1:
            detail['r_keyword'] = +1.0
        else:
            detail['r_keyword'] = min(1.0 + 0.2 * (hits - 1), 1.5)

        # ---- R_quality: 输出质量 ----
        detail['r_quality'] = self._quality_reward(text, question)

        return sum(detail.values()), detail

    def _quality_reward(self, text, question=""):
        """输出质量奖励。"""
        reward = 0.0
        chars = len(text)

        # 1. 长度检查
        if chars < 5:
            reward -= 0.3
        elif chars > 500:
            reward -= 0.1
        elif 10 <= chars <= 200:
            reward += 0.1

        # 2. 重复惩罚: 字符级 4-gram
        if chars >= 16:
            ngrams = [text[i:i+4] for i in range(chars - 3)]
            unique_ratio = len(set(ngrams)) / len(ngrams)
            if unique_ratio < 0.3:
                reward -= 0.5  # 严重重复
            elif unique_ratio < 0.5:
                reward -= 0.3
            elif unique_ratio < 0.7:
                reward -= 0.1

        # 3. 相关性: 回复中是否提到了问题中的关键名词
        if question:
            # 简单检查：问题中的实体词是否出现在回复中
            q_chars = set(question)
            overlap = sum(1 for c in text if c in q_chars and '\u4e00' <= c <= '\u9fff')
            if overlap > 3:
                reward += 0.1

        return max(reward, -1.0)  # 下限 -1.0


# ============================================================
# RL 数据集（对话式）
# ============================================================

class RLDomainDataset(Dataset):
    """加载对话式 RL 数据。

    每条数据包含:
      - prompt: ChatML 格式 prompt（到 assistant 开头）
      - keywords: 正确答案关键词列表
      - domain: 领域名
      - question: 原始问题文本
    """

    def __init__(self, data_path, tokenizer, max_prompt_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []

        ds = load_from_disk(data_path)
        if isinstance(ds, DatasetDict):
            ds = ds[list(ds.keys())[0]]

        skipped = 0
        for row in ds:
            prompt_messages = row['prompt_messages']
            keywords = row['keywords']
            domain = row.get('domain', '')

            if not keywords:
                skipped += 1
                continue

            # 构造 prompt
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) > max_prompt_length:
                skipped += 1
                continue

            # 提取原始问题
            question = ""
            for m in prompt_messages:
                if m['role'] == 'user':
                    question = m['content']

            self.data.append({
                'prompt': prompt,
                'keywords': keywords,
                'domain': domain,
                'question': question,
            })

        print(f"  RL dataset: {len(self.data)} valid / {len(ds)} total (skipped {skipped})")

        # 领域分布
        from collections import Counter
        domain_counts = Counter(d['domain'] for d in self.data)
        for dom, cnt in domain_counts.most_common():
            print(f"    {dom:<12} {cnt:>6}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# 工具函数
# ============================================================

def get_lr(step, total_steps, max_lr, warmup_steps):
    min_lr = max_lr / 10
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay))


def compute_response_log_prob(model, input_ids, response_start, device):
    """计算模型对 response 部分的 log probability。

    用 model.forward() 拿到 per-token CE loss，取反得 log_prob。
    只取 response 部分的 token。

    Args:
        model: SNN 模型
        input_ids: 完整序列 (prompt + response), shape (1, seq_len)
        response_start: response 在 input_ids 中的起始位置

    Returns:
        log_prob: response 部分的平均 log probability (scalar, 有梯度)
    """
    seq_len = input_ids.shape[1]
    if seq_len < 2 or response_start >= seq_len:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # model.forward 内部会 reset_net
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(input_ids[:, :-1], input_ids[:, 1:])

    # out.last_loss: per-token CE loss, shape (seq_len-1,)
    # CE loss = -log p(y_t | y_{<t})
    # response tokens 对应 input_ids[response_start:] 作为 targets
    # → last_loss[response_start-1:]
    loss_start = max(response_start - 1, 0)
    response_ce = out.last_loss[loss_start:]

    if response_ce.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # log_prob = -CE_loss (mean over response tokens)
    log_prob = -response_ce.mean()
    return log_prob


# ============================================================
# RL 训练步
# ============================================================

def rl_step(model, tokenizer, batch, device, reward_fn, max_gen_tokens, temperature):
    """单步 REINFORCE（对话式关键词奖励）。

    流程:
      1. 对 prompt 生成自由回复（no grad）
      2. 关键词匹配计算奖励
      3. 计算 log_prob(response | prompt)（有梯度）
      4. REINFORCE loss = -advantage * log_prob
    """
    total_reward = 0.0
    hit_count = 0
    total_count = 0
    r_keyword_sum = 0.0
    r_quality_sum = 0.0
    log_prob_reward_pairs = []
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    for item in batch:
        prompt = item['prompt']
        keywords = item['keywords']
        question = item.get('question', '')

        prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        prompt_len = prompt_ids.shape[1]

        # 生成回复
        model.eval()
        with torch.no_grad():
            gen_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_gen_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=im_end_id,
            )
        model.train()

        response_ids = gen_ids[0, prompt_len:]
        if response_ids.numel() == 0:
            total_count += 1
            total_reward -= 1.5
            continue

        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # 计算奖励
        reward, detail = reward_fn.compute(response_text, keywords, question)
        total_reward += reward
        total_count += 1
        r_keyword_sum += detail['r_keyword']
        r_quality_sum += detail['r_quality']

        if detail['r_keyword'] > 0.5:
            hit_count += 1

        # 计算 log_prob
        full_ids = gen_ids[:, :prompt_len + response_ids.numel()]
        log_prob = compute_response_log_prob(model, full_ids, prompt_len, device)
        log_prob_reward_pairs.append((log_prob, reward))

    if not log_prob_reward_pairs:
        return None, {'reward': 0, 'hit_rate': 0, 'r_keyword': 0, 'r_quality': 0}

    # REINFORCE with baseline
    mean_reward = sum(r for _, r in log_prob_reward_pairs) / len(log_prob_reward_pairs)

    loss = torch.zeros(1, device=device, requires_grad=False)
    for log_prob, reward in log_prob_reward_pairs:
        advantage = reward - mean_reward
        loss = loss + (-advantage * log_prob)
    loss = loss / len(log_prob_reward_pairs)

    if not loss.requires_grad:
        loss = sum(lp * 0.0 for lp, _ in log_prob_reward_pairs)

    n = max(total_count, 1)
    stats = {
        'reward': total_reward / n,
        'hit_rate': hit_count / n,
        'r_keyword': r_keyword_sum / n,
        'r_quality': r_quality_sum / n,
        'mean_log_prob': sum(lp.item() for lp, _ in log_prob_reward_pairs) / len(log_prob_reward_pairs),
    }
    return loss, stats


# ============================================================
# 评估
# ============================================================

def evaluate(model, tokenizer, eval_data, device, reward_fn, max_gen_tokens, n_samples=100):
    """评估当前策略的关键词命中率。"""
    model.eval()
    import random
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    indices = random.sample(range(len(eval_data)), min(n_samples, len(eval_data)))

    total_reward = 0.0
    hits = 0
    total = 0
    samples = []

    for idx in indices:
        item = eval_data[idx]
        prompt_ids = tokenizer(item['prompt'], return_tensors='pt')['input_ids'].to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_gen_tokens,
                temperature=0.1,
                top_k=10,
                eos_token_id=im_end_id,
            )

        response_ids = gen_ids[0, prompt_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        reward, detail = reward_fn.compute(
            response_text, item['keywords'], item.get('question', ''))

        total_reward += reward
        if detail['r_keyword'] > 0.5:
            hits += 1
        total += 1

        if len(samples) < 5:
            samples.append({
                'domain': item['domain'],
                'keywords': item['keywords'][:3],
                'generated': response_text[:100],
                'reward': reward,
            })

    model.train()

    hit_rate = hits / max(total, 1)
    avg_r = total_reward / max(total, 1)

    print(f"  [Eval] n={total}, hit_rate={hit_rate:.1%}, avg_reward={avg_r:+.3f}")
    for s in samples:
        tag = "O" if s['reward'] > 0.5 else "X"
        print(f"    [{tag}] [{s['domain']}] expect={s['keywords']} | gen: {s['generated'][:80]}")

    return hit_rate, avg_r


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN LM RL Training (REINFORCE)")

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=64002)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=24)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)

    # RL 参数
    parser.add_argument('--sft_ckpt', type=str, required=True,
                        help='SFT checkpoint 路径（RL 起点）')
    parser.add_argument('--data_path', type=str, default='data/neuronspark-sft')
    parser.add_argument('--max_gen_tokens', type=int, default=64,
                        help='生成回复最大 token 数')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='生成温度（需要探索性，不宜太低）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每步采样数（越大 baseline 越稳定）')

    # 训练参数
    parser.add_argument('--out_dir', type=str, default='checkpoints_rl')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=200)

    # 数据
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer/')
    parser.add_argument('--dashboard_dir', type=str, default=None)

    args = parser.parse_args()

    # ==================== 初始化 ====================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading SFT model from {args.sft_ckpt}...")
    model = SNNLanguageModel(
        vocab_size=args.vocab_size,
        D=args.D, N=args.N, K=args.K,
        num_layers=args.num_layers, D_ff=args.D_ff,
        v_th_min=args.v_th_min,
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    for name, param in model.named_parameters():
        if name.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th')):
            param.data = param.data.float()
    load_model_weights(args.sft_ckpt, model, device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {total_params / 1e6:.1f}M")

    # ==================== 奖励函数 ====================
    reward_fn = RewardFunction()

    # ==================== 优化器 ====================
    _pg = model.get_param_groups()
    _neuron_keys = ('input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron')
    neuron_params = [p for k in _neuron_keys for p in _pg.get(k, [])]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.learning_rate, 'lr_mult': 1.0},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult)},
    ])

    # ==================== 数据 ====================
    print(f"Loading RL data from {args.data_path}...")
    rl_dataset = RLDomainDataset(args.data_path, tokenizer)

    dataloader = DataLoader(
        rl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,  # 返回 list of dicts
    )

    total_steps = len(dataloader) * args.epochs
    os.makedirs(args.out_dir, exist_ok=True)

    # ==================== TensorBoard ====================
    writer = None
    if args.dashboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.dashboard_dir)

    # ==================== 训练 ====================
    print(f"\n{'='*60}")
    print(f"SNN Language Model RL Training (REINFORCE)")
    print(f"  SFT checkpoint:  {args.sft_ckpt}")
    print(f"  RL data:         {args.data_path} ({len(rl_dataset)} samples)")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Steps/epoch:     {len(dataloader)}")
    print(f"  Total steps:     {total_steps}")
    print(f"  LR:              {args.learning_rate}")
    print(f"  Max gen tokens:  {args.max_gen_tokens}")
    print(f"  Temperature:     {args.temperature}")
    print(f"{'='*60}\n")

    global_step = 0
    ema_reward = 0.0
    ema_accuracy = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            # 学习率调度
            lr = get_lr(global_step, total_steps, args.learning_rate, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * pg.get('lr_mult', 1.0)

            # RL 训练步
            loss, stats = rl_step(
                model, tokenizer, batch, device,
                reward_fn, args.max_gen_tokens, args.temperature)

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # EMA 统计
            alpha = 0.05
            ema_reward = (1 - alpha) * ema_reward + alpha * stats['reward']
            ema_accuracy = (1 - alpha) * ema_accuracy + alpha * stats['hit_rate']

            # 日志
            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"RL [{epoch+1}/{args.epochs}]({global_step}/{total_steps}) "
                    f"r:{stats['reward']:+.2f} "
                    f"[kw:{stats['r_keyword']:+.2f} q:{stats['r_quality']:+.2f}] "
                    f"hit:{stats['hit_rate']:.0%} "
                    f"ema_r:{ema_reward:+.3f} ema_hit:{ema_accuracy:.1%} "
                    f"lr:{lr:.1e} {elapsed/60:.0f}min",
                    flush=True,
                )

            # TensorBoard
            if writer:
                writer.add_scalar('rl/reward', stats['reward'], global_step)
                writer.add_scalar('rl/hit_rate', stats['hit_rate'], global_step)
                writer.add_scalar('rl/ema_reward', ema_reward, global_step)
                writer.add_scalar('rl/ema_hit_rate', ema_accuracy, global_step)
                writer.add_scalar('rl/r_keyword', stats['r_keyword'], global_step)
                writer.add_scalar('rl/r_quality', stats['r_quality'], global_step)
                writer.add_scalar('rl/lr', lr, global_step)
                if 'mean_log_prob' in stats:
                    writer.add_scalar('rl/mean_log_prob', stats['mean_log_prob'], global_step)

            # 评估
            if (global_step + 1) % args.eval_interval == 0:
                eval_acc, eval_reward = evaluate(
                    model, tokenizer, rl_dataset, device,
                    reward_fn, args.max_gen_tokens)
                if writer:
                    writer.add_scalar('rl/eval_accuracy', eval_acc, global_step)
                    writer.add_scalar('rl/eval_reward', eval_reward, global_step)

            # 保存
            if (global_step + 1) % args.save_interval == 0:
                model.eval()
                save_checkpoint(args.out_dir, model, optimizer, None,
                                global_step + 1, epoch, ema_reward, 0)
                model.train()

            global_step += 1

    # 最终保存 + 评估
    model.eval()
    save_checkpoint(args.out_dir, model, optimizer, None,
                    global_step, args.epochs, ema_reward, 0)
    print(f"\n{'='*60}")
    print("Final evaluation:")
    evaluate(model, tokenizer, rl_dataset, device, reward_fn, args.max_gen_tokens, n_samples=500)
    print(f"\nRL complete. ema_reward: {ema_reward:+.3f}, ema_accuracy: {ema_accuracy:.1%}")

    if writer:
        writer.close()
