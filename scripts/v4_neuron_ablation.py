"""v4.1 神经元设计消融: spike form (supra | quantal) × AHP (off | on).

4 配置: supra / supra_ahp / quantal / quantal_ahp. 同模型规模/数据/步数, 比 final loss + E[K] + 发放率.
默认 4 配置串行 (单 GPU); --only <name> 只跑一个 (多卡时每卡一个 + 不同 CUDA_VISIBLE_DEVICES).

数据: data/v3_pretrain_mix 的 train-00000.parquet 切片 (text 列, tokenizer_v3, 截到 SEQ).
模型: ~200M (D=512, N=16, K=12, 12 层, D_ff=1024, memory_layer_interval=4).
"""
import sys, os, time, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM

CONFIGS = {
    "supra":       dict(spike_mode="supra",   use_ahp=False),
    "supra_ahp":   dict(spike_mode="supra",   use_ahp=True,  ahp_init=0.02),
    "quantal":     dict(spike_mode="quantal", use_ahp=False),
    "quantal_ahp": dict(spike_mode="quantal", use_ahp=True,  ahp_init=0.02),
}

ap = argparse.ArgumentParser()
ap.add_argument("--only", default=None, choices=list(CONFIGS), help="只跑这一个配置 (多卡并行用)")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--n_docs", type=int, default=2400)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--out", default=None, help="日志文件 (默认 stdout)")
args = ap.parse_args()

DEV = "cuda"
DATA_PARQUET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/v3_pretrain_mix/train-00000.parquet")
TOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer_v3")
SEQ, BATCH, STEPS = args.seq, args.batch, args.steps

# ---- 数据: tokenize 切片 ----
def load_data():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    vocab = len(tok)
    texts = pq.read_table(DATA_PARQUET, columns=["text"]).slice(0, args.n_docs * 3).column("text").to_pylist()
    eos = tok.eos_token_id if tok.eos_token_id is not None else 2
    rows = []
    for tx in texts:
        ids = tok(tx, truncation=True, max_length=SEQ)["input_ids"]
        if len(ids) < 32:
            continue
        ids = ids + [eos]
        if len(ids) < SEQ:
            ids = ids + [eos] * (SEQ - len(ids))
        rows.append(ids[:SEQ])
        if len(rows) >= args.n_docs:
            break
    arr = np.array(rows, dtype=np.int64)
    return torch.from_numpy(arr).to(DEV), vocab

data_t, VOCAB = load_data()
print(f"data: {data_t.shape[0]} docs x {SEQ} tokens, vocab {VOCAB}", flush=True)
LOGF = open(args.out, "a") if args.out else None

def log(s):
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + "\n"); LOGF.flush()


def run_one(name, overrides):
    torch.manual_seed(42)
    cfg = NeuronSparkConfig(vocab_size=VOCAB, D=512, N=16, K=12, num_layers=12, D_ff=1024,
                            memory_layer_interval=4, **overrides)
    model = NeuronSparkForCausalLM(cfg).to(DEV)
    for nm, p in model.named_parameters():
        if nm.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th', '.ahp')):
            p.data = p.data.float()
        else:
            p.data = p.data.to(torch.bfloat16)
    for _, b in model.named_buffers():
        if b.is_floating_point():
            b.data = b.data.to(torch.bfloat16)
    model.train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_ahp = sum(1 for nm, _ in model.named_parameters() if nm.endswith('.ahp'))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = torch.Generator(device=DEV).manual_seed(123)
    losses = []
    t0, step0 = None, 0
    for step in range(STEPS):
        idx = torch.randint(0, data_t.shape[0], (BATCH,), generator=rng, device=DEV)
        ids = data_t[idx]
        opt.zero_grad()
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
        loss = out.loss
        if not torch.isfinite(loss):
            log(f"  [{name}] NaN/Inf at step {step}!")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step == 20:
            torch.cuda.synchronize(); t0 = time.time(); step0 = step
        if step % 200 == 0 or step == STEPS - 1:
            # 发放率: 取一层的 hidden_neuron + ffn gate
            fr_hidden = getattr(model.snn.layers[0].snn_block.hidden_neuron, '_last_firing_rate', float('nan'))
            fr_gate = getattr(model.snn.layers[0].snn_ffn.gate_neuron, '_last_firing_rate', float('nan'))
            ek = getattr(model.snn, '_output_ek', float('nan'))
            log(f"  [{name}] step {step:4d}: loss={float(loss):.4f}  E[K]_out={ek:.2f}  fire(hidden)={fr_hidden:.3f} fire(gate)={fr_gate:.3f}")
    torch.cuda.synchronize()
    spm = (time.time() - t0) / max(1, (len(losses) - 1 - step0)) * 1000 if t0 else 0
    final = sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else (sum(losses) / max(1, len(losses)))
    log(f"RESULT {name}: params={n_params:.1f}M n_ahp={n_ahp} init_loss={losses[0]:.4f} "
        f"final_loss(last100)={final:.4f} ms_step={spm:.0f}")
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return final


names = [args.only] if args.only else list(CONFIGS)
results = {}
for nm in names:
    log(f"=== {nm} {CONFIGS[nm]} ===")
    results[nm] = run_one(nm, CONFIGS[nm])

if not args.only:
    log("\n" + "=" * 60)
    base = results.get("supra", None)
    for nm in CONFIGS:
        d = (results[nm] - base) if (base is not None and nm in results) else float('nan')
        log(f"{nm:14s}  final_loss={results.get(nm, float('nan')):.4f}  Δ vs supra={d:+.4f}")
    log("=" * 60)
log("DONE")
