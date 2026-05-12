"""定位 quantal+MAL NaN 触发点的诊断脚本。

策略：
  1. 跟 ablation 同模型/数据/优化器配置 (D=512, N=16, 12 层, quantal+AHP, MAL).
  2. 每步检查 loss/params/grads finite; 每 10 步采样 per-layer 激活幅度 (res stream / V_pre / output).
  3. 前向加 hook: 检查每个 SNNDecoderLayer / SNNAttentionDecoderLayer 输出是否 finite, 第一个不 finite 的层报警.
  4. 反向加 hook: 检查每个 param 的 grad finite, 第一个 grad NaN 的 param 报警 + 它在哪一层.
  5. NaN 触发时：
     - 报告精确 step
     - 报告前向第一个 NaN tensor 在哪 (layer idx, 名字)
     - 报告反向第一个 NaN grad 在哪 (param 名字)
     - dump 关键 param 范数 (W_in/W_out per layer, v_th min/max, ahp min/max)
     - 回放最近 10 步的 per-layer 激活 max 轨迹
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from collections import defaultdict, deque
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
import neuronspark.modeling_neuronspark as _M
from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups

# ---- monkeypatch segmented PLIF to capture max|V_post| per call + backward grad magnitudes (selective only) ----
_VPOST_TRACE = []  # reset each step; collects (tag, max_abs_Vpost)
_BWD_TRACE = []    # reset each step; collects per selective-call: dict {'gout': (max,finite), 'gbeta':..., 'gu':..., 'gvth':...}
def _hook(rec, key):
    def h(g):
        try:
            rec[key] = (float(g.detach().abs().max()), bool(torch.isfinite(g).all()))
        except Exception:
            rec[key] = ("err", False)
    return h
def _wrap_seg(fn, tag):
    def wrapped(*a, **kw):
        out = fn(*a, **kw)
        try:
            vp = out[1]
            if isinstance(vp, torch.Tensor) and vp.numel():
                _VPOST_TRACE.append((tag, float(vp.detach().abs().max())))
            if tag == "sel":
                rec = {}; _BWD_TRACE.append(rec)
                # 输入: a[0]=beta_all, a[1]=u_hidden, a[2]=v_th_all  →  grad = grad_beta / grad_u / grad_vth
                for i, key in ((0, 'gbeta'), (1, 'gu'), (2, 'gvth')):
                    t = a[i] if i < len(a) else None
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        t.register_hook(_hook(rec, key))
                # 输出: out[0]=output_hidden  →  grad = g_out (从下游 W_out 等传回 SNNBlock 的)
                o0 = out[0]
                if isinstance(o0, torch.Tensor) and o0.requires_grad:
                    o0.register_hook(_hook(rec, 'gout'))
        except Exception:
            pass
        return out
    return wrapped
_M.segmented_plif_selective = _wrap_seg(_M.segmented_plif_selective, "sel")
_M.segmented_plif_rowparam = _wrap_seg(_M.segmented_plif_rowparam, "row")

# ---- monkeypatch _fused_modulation to capture per-layer β / u / v_th_hidden + 递归 Jacobian 估计 ----
# (SNNBlock.forward_parallel 调它得到 selective 神经元的 β/u/v_th; 用于看 β、电流、v_th_hidden 量级 + |J|)
_SURR_ALPHA = 4.0  # = config.surrogate_alpha 默认; 用于估 |J| = β·max(1, |1 - v_th·α/4|)
_MOD_TRACE = []  # reset each step; collects (beta_max, u_abs_max, vth_min, vth_max, Jmax_est) per SNNBlock call
def _wrap_fused(fn):
    def wrapped(*a, **kw):
        out = fn(*a, **kw)
        try:
            beta, u, v_th = out  # _fused_modulation 返回 (beta, u, v_th); v_th = v_th_min + |W_th_x(x)| 的逐 token 值
            b = beta.detach().float(); vt = v_th.detach().float()
            # 递归 Jacobian J = β·(1 - (v_th+ahp)·g_surr), g_surr ∈ [0, α/4] → J ∈ [β·(1-v_th·α/4), β]
            # |J|_max = β·max(1, |1 - v_th·α/4|)  (取 ahp=0 估计)
            jmax = (b * torch.maximum(torch.ones_like(b), torch.abs(1.0 - vt * (_SURR_ALPHA / 4.0)))).max()
            _MOD_TRACE.append((float(b.max()), float(u.detach().abs().max()), float(vt.min()), float(vt.max()), float(jmax)))
        except Exception:
            pass
        return out
    return wrapped
if hasattr(_M, "_fused_modulation"):
    _M._fused_modulation = _wrap_fused(_M._fused_modulation)

ap = argparse.ArgumentParser()
ap.add_argument("--spike_mode", default="quantal", choices=("supra", "quantal"))
ap.add_argument("--use_ahp", action="store_true", default=True)
ap.add_argument("--no_ahp", dest="use_ahp", action="store_false")
ap.add_argument("--muon_lr", type=float, default=0.02)
ap.add_argument("--lion_lr", type=float, default=1e-4)
ap.add_argument("--neuron_lr_mult", type=float, default=1.0)
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--n_docs", type=int, default=400)
ap.add_argument("--anomaly", action="store_true", help="torch.autograd.set_detect_anomaly(True) — 精确定位 backward NaN 源 op")
ap.add_argument("--anomaly_after", type=int, default=0, help="第几步起开启 anomaly mode（早期慢步跳过）")
ap.add_argument("--freeze_vth", action="store_true", help="冻结所有 *.v_th 参数（每步 opt.step 后还原）")
ap.add_argument("--freeze_beta", action="store_true", help="冻结所有 *.b_beta 参数")
ap.add_argument("--freeze_ahp", action="store_true", help="冻结所有 *.ahp 参数")
ap.add_argument("--ahp_init", type=float, default=0.02, help="AHP 初值（per-channel，use_ahp 时生效）")
ap.add_argument("--vth_reg", type=float, default=0.0, help="PLIFNode.v_th 朝 init 的二次正则权重（0=关）")
args = ap.parse_args()

DEV = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BIN = os.path.join(ROOT, "data/sft_think_binned_2048")

# ----- data -----
if os.path.isdir(DATA_BIN):
    bins = sorted([f for f in os.listdir(DATA_BIN) if f.endswith(".bin") and ".mask" not in f])
    arr = np.fromfile(os.path.join(DATA_BIN, bins[0]), dtype=np.uint32).reshape(-1, 2048)[:args.n_docs]
    arr = arr[:, :args.seq].astype(np.int64)
    VOCAB = 128387
else:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer_v3"))
    VOCAB = len(tok)
    texts = pq.read_table(os.path.join(ROOT, "data/v3_pretrain_mix/train-00000.parquet"),
                          columns=["text"]).slice(0, args.n_docs * 3).column("text").to_pylist()
    eos = tok.eos_token_id or 2
    rows = []
    for tx in texts:
        ids = tok(tx, truncation=True, max_length=args.seq)["input_ids"]
        if len(ids) < 32: continue
        ids = (ids + [eos])[:args.seq]
        if len(ids) < args.seq: ids = ids + [eos]*(args.seq-len(ids))
        rows.append(ids)
        if len(rows) >= args.n_docs: break
    arr = np.array(rows, dtype=np.int64)
data_t = torch.from_numpy(arr).to(DEV)
print(f"data: {data_t.shape}, vocab {VOCAB}", flush=True)

# ----- model -----
torch.manual_seed(42)
cfg = NeuronSparkConfig(vocab_size=VOCAB, D=512, N=16, K=12, num_layers=12, D_ff=1024,
                        memory_layer_interval=4,
                        spike_mode=args.spike_mode, use_ahp=args.use_ahp, ahp_init=args.ahp_init,
                        v_th_reg_weight=args.vth_reg)
model = NeuronSparkForCausalLM(cfg).to(DEV)
for nm, p in model.named_parameters():
    if nm.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th', '.ahp')):
        p.data = p.data.float()
    else:
        p.data = p.data.to(torch.bfloat16)
model.train()

# ----- freeze probe: snapshot params we'll restore after every opt.step() -----
_frozen = {}
for nm, p in model.named_parameters():
    if (args.freeze_vth and nm.endswith('.v_th')) or (args.freeze_beta and nm.endswith('.b_beta')) or (args.freeze_ahp and nm.endswith('.ahp')):
        _frozen[nm] = p.data.clone()
if _frozen:
    print(f"[freeze] {len(_frozen)} param tensors frozen (will restore after each opt.step): "
          f"vth={args.freeze_vth} beta={args.freeze_beta}", flush=True)

# ----- optimizer -----
groups = build_muon_adam_lion_param_groups(model, muon_lr=args.muon_lr, adam_base_lr=2e-4,
                                            lion_lr=args.lion_lr, neuron_lr_mult=args.neuron_lr_mult)
opt = SingleDeviceMoonshotMuonAdamLion(groups)

# ----- forward NaN-detection hooks: per-layer output -----
fwd_history = deque(maxlen=20)  # 最近 20 步的 per-layer 激活幅度
first_nan = {"step": None, "where": None, "details": None}

def make_layer_hook(layer_idx, layer_name):
    def hook(module, inp, out):
        # output of SNNDecoderLayer.forward_parallel: (h, ponder_cost). h shape (T, b, D).
        # if it's a tuple, look at h
        if isinstance(out, tuple):
            h = out[0]
            extras = []
            for o in out[1:]:
                if isinstance(o, torch.Tensor) and o.numel() < 100:
                    extras.append((o.detach().float().abs().max().item(), o.detach().float().abs().min().item()))
            if h is None or not isinstance(h, torch.Tensor):
                return
        else:
            h = out
        if not torch.isfinite(h).all():
            if first_nan["where"] is None:
                first_nan["where"] = f"layer{layer_idx}[{layer_name}].forward_out"
                first_nan["details"] = f"shape={tuple(h.shape)}, has_nan={h.isnan().any().item()}, has_inf={h.isinf().any().item()}"
        return out
    return hook

handles = []
for i, layer in enumerate(model.snn.layers):
    h = layer.register_forward_hook(make_layer_hook(i, type(layer).__name__))
    handles.append(h)

# also hook output_neuron, decode_proj
def output_hook(name):
    def hook(module, inp, out):
        x = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)
        if x is not None and not torch.isfinite(x).all() and first_nan["where"] is None:
            first_nan["where"] = name
            first_nan["details"] = f"shape={tuple(x.shape)} nan={x.isnan().any().item()} inf={x.isinf().any().item()}"
    return hook
handles.append(model.snn.output_neuron.register_forward_hook(output_hook("output_neuron")))
handles.append(model.snn.decode_proj.register_forward_hook(output_hook("decode_proj")))


def dump_state(step):
    print(f"\n=== DUMP at step {step} ===", flush=True)
    print(f"[first NaN where] {first_nan.get('where')} :: {first_nan.get('details')}", flush=True)
    # param norms
    print("\n[param norms (top 20 by name, max abs)]")
    rows = []
    for nm, p in model.named_parameters():
        if p.numel() == 0: continue
        finite = torch.isfinite(p.data).all().item()
        mx = p.data.abs().max().item() if finite else float('inf')
        mn = p.data.min().item() if finite else float('inf')
        rows.append((nm, finite, mx, mn, p.data.float().norm().item()))
    rows.sort(key=lambda r: -r[2])
    for nm, fin, mx, mn, n in rows[:20]:
        print(f"  {nm:80s} finite={fin} max|{mx:.4e}|  min={mn:+.4e}  norm={n:.4e}", flush=True)
    # grad norms
    print("\n[grad norms (top 20 by max abs)]")
    grows = []
    for nm, p in model.named_parameters():
        if p.grad is None: continue
        finite = torch.isfinite(p.grad).all().item()
        mx = p.grad.abs().max().item() if finite else float('inf')
        n = p.grad.float().norm().item() if finite else float('inf')
        grows.append((nm, finite, mx, n))
    grows.sort(key=lambda r: -r[2])
    for nm, fin, mx, n in grows[:20]:
        print(f"  {nm:80s} grad finite={fin} max|{mx:.4e}|  norm={n:.4e}", flush=True)
    # firing rate per layer (hidden + ffn gate)
    print("\n[per-layer firing rate]")
    for i, layer in enumerate(model.snn.layers):
        h_fr = getattr(getattr(layer, 'snn_block', None), 'hidden_neuron', None)
        h_fr = getattr(h_fr, '_last_firing_rate', None) if h_fr else None
        g_fr = getattr(getattr(layer, 'snn_ffn', None), 'gate_neuron', None)
        g_fr = getattr(g_fr, '_last_firing_rate', None) if g_fr else None
        print(f"  layer{i:2d} ({type(layer).__name__:25s}): hidden={h_fr}  ffn_gate={g_fr}", flush=True)
    # per-call snapshot from THIS (failing) forward
    print(f"\n[this-forward per-SNNBlock-layer (β.max, u.abs.max, v_th_hidden [min,max], |J|_max≈β·max(1,|1−v_th·α/4|)) — order = layer order]")
    for i, t in enumerate(_MOD_TRACE):
        bm, um, vmn, vmx, jm = t
        print(f"  snnblock-call#{i}: β_max={bm:.6f}  u_abs_max={um:.4e}  v_th_hidden=[{vmn:.4e},{vmx:.4e}]  |J|_max={jm:.4e}", flush=True)
    print(f"[this-forward per segmented-PLIF call (tag, max|V_post|)]")
    for tag, vp in _VPOST_TRACE:
        print(f"  {tag}: max|V_post|={vp:.4e}", flush=True)
    print(f"[backward grad magnitudes per selective(SNNBlock-hidden) call — gout=∂L/∂out 从下游传回, gbeta/gu/gvth=∂L/∂(β,u,v_th)]")
    for i, rec_ in enumerate(_BWD_TRACE):
        parts = []
        for k in ('gout', 'gbeta', 'gu', 'gvth'):
            if k in rec_:
                mx, fin = rec_[k]
                parts.append(f"{k}={mx if isinstance(mx,str) else f'{mx:.3e}'}{'' if fin else '(INF!)'}")
            else:
                parts.append(f"{k}=?")
        print(f"  sel-call#{i}: " + "  ".join(parts), flush=True)
    # recent activation max trajectory
    print(f"\n[recent trajectory (last {len(fwd_history)} steps): loss / Vpost_max / β_max / u_max / vthH_min / hidden_fr_μ]")
    for s in fwd_history:
        line = f"  step {s['step']:4d}: loss={s['loss']:.4f}"
        if 'res_max' in s: line += f"  Vpost_max={s['res_max']:.3e}"
        if 'beta_max' in s: line += f"  β_max={s['beta_max']:.6f}"
        if 'u_max' in s: line += f"  u_max={s['u_max']:.3e}"
        if 'vthH_min' in s: line += f"  vthH_min={s['vthH_min']:.3e}"
        if 'fr_h' in s: line += f"  hidden_fr_μ={s['fr_h']:.4f}"
        print(line, flush=True)


# ----- train loop with NaN detection -----
rng = torch.Generator(device=DEV).manual_seed(123)
print(f"\nstart diag run: spike={args.spike_mode}, use_ahp={args.use_ahp}, muon_lr={args.muon_lr}, lion_lr={args.lion_lr}, max_steps={args.max_steps}", flush=True)

for step in range(args.max_steps):
    if args.anomaly and step == args.anomaly_after:
        torch.autograd.set_detect_anomaly(True)
        print(f"[anomaly] set_detect_anomaly(True) at step {step}", flush=True)
    idx = torch.randint(0, data_t.shape[0], (args.batch,), generator=rng, device=DEV)
    ids = data_t[idx]
    opt.zero_grad()
    first_nan["where"] = None
    _VPOST_TRACE.clear(); _MOD_TRACE.clear(); _BWD_TRACE.clear()
    with torch.amp.autocast(DEV, dtype=torch.bfloat16):
        out = model(input_ids=ids, labels=ids)
    loss = out.loss
    rec = {"step": step, "loss": float(loss) if torch.isfinite(loss) else float('nan')}
    # membrane runaway probe: max|V_post| over all segmented-PLIF calls this fwd
    if _VPOST_TRACE:
        rec["vpost_max"] = max(v for _, v in _VPOST_TRACE)
        rec["res_max"] = rec["vpost_max"]
    # selective β / u / v_th_hidden / |J| probe (per SNNBlock layer, from _fused_modulation)
    if _MOD_TRACE:
        rec["beta_max"] = max(t[0] for t in _MOD_TRACE)
        rec["u_max"] = max(t[1] for t in _MOD_TRACE)
        rec["vthH_min"] = min(t[2] for t in _MOD_TRACE)
        rec["vthH_max"] = max(t[3] for t in _MOD_TRACE)
        rec["Jmax"] = max(t[4] for t in _MOD_TRACE)
    fr_h_list = []
    for layer in model.snn.layers:
        nb = getattr(layer, 'snn_block', None)
        if nb is not None:
            fr = getattr(nb.hidden_neuron, '_last_firing_rate', None)
            if fr is not None: fr_h_list.append(fr)
    if fr_h_list: rec["fr_h"] = sum(fr_h_list) / len(fr_h_list)
    fwd_history.append(rec)

    if not torch.isfinite(loss):
        print(f"\n*** NaN/Inf in loss at step {step} ***", flush=True)
        dump_state(step)
        break
    loss.backward()
    # check grads
    any_nan_grad = False
    nan_grad_name = None
    for nm, p in model.named_parameters():
        if p.grad is None: continue
        if not torch.isfinite(p.grad).all():
            any_nan_grad = True
            nan_grad_name = nm
            break
    if any_nan_grad:
        print(f"\n*** NaN/Inf in grad at step {step}, first nan param: {nan_grad_name} ***", flush=True)
        dump_state(step)
        break

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    # restore frozen params (freeze probe)
    if _frozen:
        for nm, p in model.named_parameters():
            if nm in _frozen:
                p.data.copy_(_frozen[nm])
    # check params after step
    any_nan_param = False
    nan_param_name = None
    for nm, p in model.named_parameters():
        if not torch.isfinite(p.data).all():
            any_nan_param = True
            nan_param_name = nm
            break
    if any_nan_param:
        print(f"\n*** NaN/Inf in PARAM after opt.step() at step {step}, first nan param: {nan_param_name} ***", flush=True)
        dump_state(step)
        break

    if step % 10 == 0 or step in (50, 100, 150, 200):
        # 取一些代表性的诊断数字
        v_th_min = min(p.data.min().item() for nm,p in model.named_parameters() if nm.endswith('.v_th'))
        v_th_max = max(p.data.max().item() for nm,p in model.named_parameters() if nm.endswith('.v_th'))
        ahp_vals = [p.data.float() for nm,p in model.named_parameters() if nm.endswith('.ahp')]
        ahp_str = f"ahp[{min(a.min().item() for a in ahp_vals):.4f}, {max(a.max().item() for a in ahp_vals):.4f}]" if ahp_vals else "ahp=N/A"
        win_max = max(p.data.float().abs().max().item() for nm,p in model.named_parameters() if 'W_in' in nm)
        wout_max = max(p.data.float().abs().max().item() for nm,p in model.named_parameters() if 'W_out' in nm)
        bb = [p.data.float() for nm,p in model.named_parameters() if nm.endswith('.b_beta')]
        if bb:
            bb_max = max(b.max().item() for b in bb); bb_min = min(b.min().item() for b in bb)
            import math as _m
            beta_max = 1.0/(1.0+_m.exp(-bb_max)); beta_min = 1.0/(1.0+_m.exp(-bb_min))
            bb_str = f"b_beta[{bb_min:.2f},{bb_max:.2f}]→β[{beta_min:.4f},{beta_max:.5f}]"
        else:
            bb_str = "b_beta=N/A"
        gnorm = sum((p.grad.float().norm()**2).item() for p in model.parameters() if p.grad is not None) ** 0.5
        fr_h = rec.get("fr_h", float('nan'))
        vpm = rec.get("vpost_max", float('nan'))
        bmx = rec.get("beta_max", float('nan')); umx = rec.get("u_max", float('nan')); vhmn = rec.get("vthH_min", float('nan'))
        vhmx = rec.get("vthH_max", float('nan')); jmx = rec.get("Jmax", float('nan'))
        # W_beta_x norm: bias 重构后 β=sigmoid(W_beta_x(x)), W_beta_x 涨→β→1
        wbx_max = max((p.data.float().abs().max().item() for nm,p in model.named_parameters() if 'W_beta_x' in nm), default=float('nan'))
        print(f"step {step:4d}: loss={float(loss):.4f}  gnorm={gnorm:.2e}  Vpost_max={vpm:.3e}  β_max={bmx:.6f}  u_max={umx:.3e}  vthH=[{vhmn:.3e},{vhmx:.3e}]  |J|_max={jmx:.3e}  v_th[{v_th_min:.4f},{v_th_max:.4f}]  W_th_x_max={max((p.data.float().abs().max().item() for nm,p in model.named_parameters() if 'W_th_x' in nm), default=float('nan')):.3e}  W_in_max={win_max:.3e}  W_beta_x_max={wbx_max:.3e}  fr_h={fr_h:.4f}", flush=True)

print("\n=== DIAG END ===", flush=True)
for h in handles: h.remove()
