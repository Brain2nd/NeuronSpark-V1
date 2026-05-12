"""测: MAL (Muon+Adam+Lion) 优化器下, 神经元参数 (.w/.v_th/.ahp) 用 bf16 是否能正常训练?
背景: 原本神经元参数保 fp32 是因为 Adam 下 bf16 的微小更新被量化掉 + sigmoid/softplus 在 bf16 饱和 → 这些参数得不到训练.
MAL 里这些参数走 Lion (lr·sign(grad), 定步长 ~lion_lr), 可能不受"微小更新被量化"影响 —— 验证.

跑 3 个变体, 各训 N 步, 比 (a) 神经元参数相对 init 的变化量 (≈0 = 没训) (b) loss 趋势 (c) NaN:
  A. full-bf16 + MAL    —— 所有参数 bf16, Lion 管神经元参数
  B. mixed   + MAL      —— bf16 矩阵 + fp32 神经元 (当前默认)
  C. full-bf16 + Adam   —— 对照: 印证"Adam 下 bf16 神经元不训" (原始动机)

用法: CUDA_VISIBLE_DEVICES=N python scripts/v4_test_bf16_neurons.py [--steps 1000 --D 128 --N 8 --num_layers 6]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--D", type=int, default=128)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--K", type=int, default=6)
ap.add_argument("--num_layers", type=int, default=6)
ap.add_argument("--D_ff", type=int, default=256)
ap.add_argument("--vocab", type=int, default=256)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--muon_lr", type=float, default=0.005)
ap.add_argument("--lion_lr", type=float, default=1e-4)
ap.add_argument("--use_ahp", action="store_true")
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEURON_SUFFIXES = ('.w', '.v_th', '.ahp')


def build(full_bf16):
    torch.manual_seed(42)
    cfg = NeuronSparkConfig(vocab_size=args.vocab, D=args.D, N=args.N, K=args.K, num_layers=args.num_layers,
                            D_ff=args.D_ff, memory_layer_interval=2, spike_mode="quantal", use_ahp=args.use_ahp)
    m = NeuronSparkForCausalLM(cfg).to(DEV)
    for n, p in m.named_parameters():
        if (not full_bf16) and n.endswith(NEURON_SUFFIXES):
            p.data = p.data.float()
        else:
            p.data = p.data.to(torch.bfloat16)
    return m


def neuron_snapshot(m):
    return {n: p.data.detach().float().clone() for n, p in m.named_parameters() if n.endswith(NEURON_SUFFIXES)}


def rel_change(m, snap):
    """每个神经元参数张量相对 init 的 ‖Δ‖/‖init‖, 返回 (按名聚合的 min/median/max)."""
    rs = []
    for n, p in m.named_parameters():
        if n.endswith(NEURON_SUFFIXES):
            init = snap[n]; cur = p.data.detach().float()
            denom = init.norm().item() + 1e-12
            rs.append((n, (cur - init).norm().item() / denom))
    vals = sorted(r for _, r in rs)
    return vals[0], vals[len(vals) // 2], vals[-1], len(vals)


def run(name, full_bf16, opt_kind):
    m = build(full_bf16); m.train()
    snap = neuron_snapshot(m)
    if opt_kind == "mal":
        from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups
        groups = build_muon_adam_lion_param_groups(m, muon_lr=args.muon_lr, adam_base_lr=2e-4,
                                                    lion_lr=args.lion_lr, neuron_lr_mult=1.0, weight_decay_muon=0.01)
        opt = SingleDeviceMoonshotMuonAdamLion(groups)
    else:  # adam
        groups = [{"params": [p for n, p in m.named_parameters() if not n.endswith(NEURON_SUFFIXES)], "lr": 2e-4},
                  {"params": [p for n, p in m.named_parameters() if n.endswith(NEURON_SUFFIXES)], "lr": 2e-3}]  # neuron lr×10
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = torch.Generator(device=DEV).manual_seed(123)
    losses, n_nan = [], 0
    for step in range(args.steps):
        ids = torch.randint(0, args.vocab, (args.batch, args.seq), generator=rng, device=DEV)
        opt.zero_grad()
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            out = m(input_ids=ids, labels=ids)
        loss = out.loss
        if not torch.isfinite(loss):
            n_nan += 1
            if n_nan > 20: break
            continue
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):
            n_nan += 1; opt.zero_grad(); continue
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    lo, med, hi, ntens = rel_change(m, snap)
    # 单独看 .v_th 和 .w
    def _rc_filter(suf):
        rs = []
        for n, p in m.named_parameters():
            if n.endswith(suf):
                init = snap[n]
                rs.append((p.data.detach().float() - init).norm().item() / (init.norm().item() + 1e-12))
        return (min(rs), sorted(rs)[len(rs)//2], max(rs)) if rs else (0, 0, 0)
    vth_rc = _rc_filter('.v_th'); w_rc = _rc_filter('.w')
    init_loss = losses[0] if losses else float('nan')
    final_loss = sum(losses[-50:]) / len(losses[-50:]) if len(losses) >= 50 else (sum(losses)/max(1,len(losses)))
    print(f"\n=== {name} ===")
    print(f"  loss: {init_loss:.4f} → {final_loss:.4f}  (Δ={init_loss - final_loss:+.4f})   n_nan_steps={n_nan}")
    print(f"  神经元参数相对 init 变化 ‖Δ‖/‖init‖ ({ntens} 张量): min={lo:.2e} median={med:.2e} max={hi:.2e}")
    print(f"    其中 .v_th: min={vth_rc[0]:.2e} median={vth_rc[1]:.2e} max={vth_rc[2]:.2e}")
    print(f"    其中 .w   : min={w_rc[0]:.2e} median={w_rc[1]:.2e} max={w_rc[2]:.2e}")
    print(f"  → 神经元参数 {'有训' if med > 1e-3 else '基本没动 (没训)'}")
    del m, opt; import gc; gc.collect(); torch.cuda.empty_cache()


print(f"config: D={args.D} N={args.N} K={args.K} layers={args.num_layers} vocab={args.vocab} steps={args.steps} muon_lr={args.muon_lr} lion_lr={args.lion_lr} use_ahp={args.use_ahp}")
run("A. full-bf16 + MAL", full_bf16=True, opt_kind="mal")
run("B. mixed (fp32 neuron) + MAL [当前默认]", full_bf16=False, opt_kind="mal")
run("C. full-bf16 + Adam [对照: 原始动机]", full_bf16=True, opt_kind="adam")
print("\n=== DONE ===")
