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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--only", default=None, choices=list(CONFIGS), help="只跑这一个配置 (多卡并行用)")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--n_docs", type=int, default=2400)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--data", default=None,
                help="数据源: .parquet (text 列, 用 tokenizer_v3 tokenize) 或 含 *.bin 的目录 (已 tokenize 的 uint32, reshape 到 --binned_len). 默认自动找.")
ap.add_argument("--binned_len", type=int, default=2048, help="binned .bin 的行长 (sft_think_binned_2048=2048)")
ap.add_argument("--vocab", type=int, default=None, help="binned 数据时手动指定 vocab (默认 128387)")
ap.add_argument("--out", default=None, help="日志文件 (默认 stdout)")
ap.add_argument("--optimizer", default="adam", choices=("adam", "muon_adam_lion"),
                help="adam (baseline, lr_mult=10 for neurons) | "
                     "muon_adam_lion (Muon for matrices + Adam for embed/norm + Lion for 逐通道神经元参数 .w/.v_th/.ahp (1D tensor); DeepSpeed-ZeRO0 兼容)")
ap.add_argument("--muon_lr", type=float, default=0.02)
ap.add_argument("--adam_base_lr", type=float, default=2e-4)
ap.add_argument("--lion_lr", type=float, default=5e-4)
ap.add_argument("--neuron_lr_mult", type=float, default=10.0)
ap.add_argument("--save_to", default=None, help="保存 {state_dict, config, ...} 到此路径 (用于后续分析)")
ap.add_argument("--save_every", type=int, default=500, help="每 N 步存一次 ckpt (覆盖 --save_to); 0=只在结束时存. 防 run 被杀后全丢.")
ap.add_argument("--lsuv", action="store_true", help="训练前跑 LSUV v2 初始化 (校准 v_th + scale W_in 让 hidden 发放 ~lsuv_target_p)")
ap.add_argument("--lsuv_target_p", type=float, default=0.3)
# 模型尺寸（默认 = 之前 ablation 的 ~290M; 调大跑更大规模 derisk）
ap.add_argument("--D", type=int, default=512)
ap.add_argument("--N", type=int, default=16)
ap.add_argument("--K_max", type=int, default=12)
ap.add_argument("--num_layers", type=int, default=12)
ap.add_argument("--D_ff", type=int, default=1024)
ap.add_argument("--memory_layer_interval", type=int, default=4)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--muon_wd", type=float, default=0.0, help="weight decay on the Muon (matrix) group — bounds matrix growth, helps quantal stability")
ap.add_argument("--vth_reg", type=float, default=0.0, help="PLIFNode.v_th 朝-init 二次正则权重 (实验 A; 实测无用/有害, 默认 0=关)")
ap.add_argument("--ahp_init", type=float, default=None, help="覆写 AHP 初值 (仅 use_ahp 配置生效; None=用 CONFIGS 默认 0.02)")
args = ap.parse_args()

DEV = "cuda"
SEQ, BATCH, STEPS = args.seq, args.batch, args.steps

# ---- 数据 ----
def _resolve_data():
    if args.data:
        return args.data
    # 自动: 优先本地 parquet, 否则找 binned 目录 (4090 上有 sft_think_binned_2048)
    p = os.path.join(_ROOT, "data/v3_pretrain_mix/train-00000.parquet")
    if os.path.exists(p):
        return p
    for d in ("data/sft_think_binned_2048", "data/sft_v3_binned_2048"):
        dd = os.path.join(_ROOT, d)
        if os.path.isdir(dd) and any(f.endswith(".bin") for f in os.listdir(dd)):
            return dd
    raise FileNotFoundError("找不到数据; 用 --data 指定 .parquet 或 binned 目录")


def load_data():
    src = _resolve_data()
    if src.endswith(".parquet"):
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(_ROOT, "tokenizer_v3"))
        vocab = len(tok)
        texts = pq.read_table(src, columns=["text"]).slice(0, args.n_docs * 3).column("text").to_pylist()
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
        print(f"data: parquet {src}", flush=True)
        return torch.from_numpy(arr).to(DEV), vocab
    # binned 目录: 已 tokenize 的 uint32, 取第一个 .bin shard, reshape(-1, binned_len), 截到 SEQ
    bins = sorted(f for f in os.listdir(src) if f.endswith(".bin") and ".mask" not in f)
    arr = np.fromfile(os.path.join(src, bins[0]), dtype=np.uint32).reshape(-1, args.binned_len)[:args.n_docs]
    arr = arr[:, :SEQ].astype(np.int64)
    vocab = args.vocab or 128387
    print(f"data: binned {src}/{bins[0]}", flush=True)
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
    cfg_kwargs = dict(overrides)
    if args.vth_reg:
        cfg_kwargs["v_th_reg_weight"] = args.vth_reg
    if args.ahp_init is not None and cfg_kwargs.get("use_ahp"):
        cfg_kwargs["ahp_init"] = args.ahp_init
    cfg = NeuronSparkConfig(vocab_size=VOCAB, D=args.D, N=args.N, K=args.K_max, num_layers=args.num_layers,
                            D_ff=args.D_ff, memory_layer_interval=args.memory_layer_interval, **cfg_kwargs)
    model = NeuronSparkForCausalLM(cfg).to(DEV)
    # 全模型 bf16 (含神经元参数 .w/.v_th/.ahp —— MAL 的 stochastic rounding 保证小更新累积); k_predictor EMA buffer 保 fp32
    for nm, p in model.named_parameters():
        p.data = p.data.to(torch.bfloat16)
    for nm, b in model.named_buffers():
        if b.is_floating_point() and 'k_predictor' not in nm:
            b.data = b.data.to(torch.bfloat16)
    if args.lsuv:
        from utils.lsuv_snn_init import lsuv_snn_init
        lsuv_snn_init(model, data_t[:2, :min(SEQ, 128)], target_p_fire=args.lsuv_target_p, n_passes=3, verbose=False)
        log(f"  [{name}] LSUV v2 init applied (target p_fire={args.lsuv_target_p})")
    model.train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_ahp = sum(1 for nm, _ in model.named_parameters() if nm.endswith('.ahp'))
    if args.optimizer == "muon_adam_lion":
        from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups
        groups = build_muon_adam_lion_param_groups(
            model, muon_lr=args.muon_lr, adam_base_lr=args.adam_base_lr,
            lion_lr=args.lion_lr, neuron_lr_mult=args.neuron_lr_mult,
            weight_decay_muon=args.muon_wd,
        )
        opt = SingleDeviceMoonshotMuonAdamLion(groups)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = torch.Generator(device=DEV).manual_seed(123)
    losses = []
    n_skipped = 0
    t0, step0 = None, 0
    ckpt_path = (args.save_to if args.only else f"{args.save_to}.{name}") if args.save_to else None

    def _save(step_, tag):
        if not ckpt_path:
            return
        cur = sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else (sum(losses) / max(1, len(losses)) if losses else float('nan'))
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": cfg.to_dict(), "name": name, "final_loss": cur, "step": step_}, ckpt_path)
        log(f"  [{name}] {tag} ckpt → {ckpt_path} (step {step_}, loss(last100)={cur:.4f})")

    for step in range(STEPS):
        idx = torch.randint(0, data_t.shape[0], (BATCH,), generator=rng, device=DEV)
        ids = data_t[idx]
        opt.zero_grad()
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
        loss = out.loss
        if not torch.isfinite(loss):
            n_skipped += 1
            log(f"  [{name}] non-finite loss at step {step} — skipping (skipped={n_skipped})")
            if n_skipped > max(50, STEPS // 50):
                log(f"  [{name}] too many non-finite steps ({n_skipped}) → aborting"); break
            continue
        loss.backward()
        # skip step if any grad non-finite
        bad_grad = any((p.grad is not None and not torch.isfinite(p.grad).all()) for p in model.parameters())
        if bad_grad:
            n_skipped += 1
            log(f"  [{name}] non-finite grad at step {step} — skipping (skipped={n_skipped})")
            opt.zero_grad(); continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        # NaN poison check: if a param went non-finite after the step, restore from no-op (clamp not possible → just zero the bad param's grad already done; here we just warn — rare)
        losses.append(float(loss))
        if step == 20:
            torch.cuda.synchronize(); t0 = time.time(); step0 = step
        if step % 200 == 0 or step == STEPS - 1:
            # 发放率: 所有层的 hidden_neuron (min~max·μ) + 第 0 层 gate; + 输出 E[K]
            fr_h = [getattr(l.snn_block.hidden_neuron, '_last_firing_rate', float('nan'))
                    for l in model.snn.layers if hasattr(l, 'snn_block')]
            fr_h = [x for x in fr_h if x == x]  # drop nan
            hsum = (f"{min(fr_h):.3f}~{max(fr_h):.3f}μ{sum(fr_h)/len(fr_h):.3f}" if fr_h else "?")
            fr_gate = getattr(model.snn.layers[0].snn_ffn.gate_neuron, '_last_firing_rate', float('nan'))
            ek = getattr(model.snn, '_output_ek', float('nan'))
            log(f"  [{name}] step {step:4d}: loss={float(loss):.4f}  E[K]_out={ek:.2f}  fire(hidden)={hsum} fire(gate0)={fr_gate:.3f}")
        if args.save_every and step > 0 and step % args.save_every == 0:
            _save(step, "periodic")
    torch.cuda.synchronize()
    spm = (time.time() - t0) / max(1, (len(losses) - 1 - step0)) * 1000 if t0 else 0
    final = sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else (sum(losses) / max(1, len(losses)))
    log(f"RESULT {name}: params={n_params:.1f}M n_ahp={n_ahp} init_loss={losses[0]:.4f} "
        f"final_loss(last100)={final:.4f} ms_step={spm:.0f}")
    _save(len(losses), "final")
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
