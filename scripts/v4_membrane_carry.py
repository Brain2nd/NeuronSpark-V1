"""V4.1 A5 — 膜电位残留 / K-frame 携带统计 (quantal vs supra 的核心力学差异，逐层逐通道).

为什么要这个: §3.8 已经从 BPTT Jacobian 角度解释了 quantal 的稳定性边界；§4b 的 E3 显示 quantal 的
长程因果影响 ~20× supra。本脚本直接量「膜里到底留了多少历史」——

  supra:   output = relu(V_pre - v_th)，V_post = min(V_pre, v_th)  → 放电后膜被硬截到 v_th，超阈部分丢弃。
  quantal: output = v_th·𝟙[V_pre>v_th]，V_post = V_pre - (v_th+ahp)·𝟙[…] → 放电后只扣掉 1 个量子 v_th(+ahp)，
           超阈余量 (V_pre - v_th - ahp) 留在膜里继续往下传 → 膜本身就是历史累加器。

逐层(SNNBlock.hidden_neuron)报:
  p_fire         : 放电率 = mean(output != 0)
  |V_pre| 分位   : 1/50/90% (V_pre = β·V_post_{prev} + u, 用捕获的 β,u 重建)
  v_th 均值      : 选择性阈值 (v_th = v_th_min + |W_th_x·x|, 逐 token 变)
  retain = V_post/v_th @fire : 放电后剩余膜 / 阈值. supra 恒 =1.0；quantal = (V_pre-v_th-ahp)/v_th 可 >1
                              → 报 mean / p90 / frac(retain>0.5) / frac(retain>1.0「放电后仍超阈」)
  carry K-frame  : token 内 |V_post[K-1]| / |V_post[0]|  (token 内 K 帧后膜的存活比)
  carry token→   : |v_carry| (传给下个 token 的膜) / token 内 |V_post| 均值

用法: CUDA_VISIBLE_DEVICES=N python scripts/v4_membrane_carry.py --ckpt path/to/ckpt.pt [--seq 512 --batch 2]
      无 ckpt → fresh-init (只看初始化力学).  对照: 分别跑 derisk_q6_bf16_5k.pt 与 derisk_s6_bf16_5k.pt.
"""
import sys, os, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from neuronspark.modeling_neuronspark import PLIFNode, SNNDecoderLayer, functional
import neuronspark.modeling_neuronspark as nm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=2)
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu")
        cfg = NeuronSparkConfig(**ck["config"])
        m = NeuronSparkForCausalLM(cfg)
        m.load_state_dict(ck["state_dict"])
        print(f"loaded {args.ckpt}: name={ck.get('name')} spike_mode={getattr(cfg,'spike_mode','?')} "
              f"use_ahp={getattr(cfg,'use_ahp','?')} step={ck.get('step')} final_loss={ck.get('final_loss')}")
    else:
        torch.manual_seed(42)
        cfg = NeuronSparkConfig(vocab_size=128387, D=512, N=16, K=12, num_layers=12, D_ff=1024,
                                memory_layer_interval=4, spike_mode="quantal", use_ahp=False, ahp_init=0.02)
        m = NeuronSparkForCausalLM(cfg)
        print(f"no ckpt → fresh-init (spike_mode={cfg.spike_mode})")
    for _, p in m.named_parameters():
        p.data = p.data.to(torch.bfloat16)
    return m.to(DEV).eval(), cfg


def get_val_batch(seq, batch, vocab):
    db = os.path.join(ROOT, "data/sft_think_binned_2048")
    if os.path.isdir(db):
        bins = sorted([f for f in os.listdir(db) if f.endswith(".bin") and ".mask" not in f])
        if bins:
            arr = np.fromfile(os.path.join(db, bins[0]), dtype=np.uint32).reshape(-1, 2048)
            arr = arr[1000:1000 + batch, :seq].astype(np.int64)
            return torch.from_numpy(arr).to(DEV)
    return torch.randint(0, vocab, (batch, seq), device=DEV)


def q(x, ps=(1, 50, 90)):
    x = np.asarray(x, dtype=np.float64)
    return tuple(float(np.percentile(x, p)) for p in ps)


def capture(model, ids, spike_mode):
    """hook segmented_plif_selective: 拿每个 SNNBlock.hidden_neuron 的 (beta,u,v_th,v_init,k_t,K) + (output,V_post,v_carry)."""
    snn = model.snn
    sel_layers = [li for li, l in enumerate(snn.layers) if isinstance(l, SNNDecoderLayer)]
    orig = nm.segmented_plif_selective
    grabbed = {}
    counter = [0]

    def _np(x, ref=None):
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
        # scalar (e.g. v_init=0.0 on first forward) → broadcast to ref's (b,H)
        return np.full((ref.shape[-2], ref.shape[-1]), float(x), dtype=np.float64) if ref is not None else float(x)

    def wrap(beta, u, v_th, v_init, k_t, K, training=True, ahp_row=None, alpha=4.0, spike_mode=spike_mode):
        out = orig(beta, u, v_th, v_init, k_t, K, training=training, ahp_row=ahp_row, alpha=alpha, spike_mode=spike_mode)
        output, V_post, v_carry = out
        li = sel_layers[counter[0] % len(sel_layers)]
        if li in grabbed:  # checkpoint recompute / repeated call — keep first
            counter[0] += 1; return out
        grabbed[li] = dict(
            beta=_np(beta), u=_np(u), v_th=_np(v_th), v_init=_np(v_init, ref=v_carry),
            k_t=k_t.detach().cpu().numpy() if isinstance(k_t, torch.Tensor) else np.asarray(k_t), K=K,
            output=_np(output), V_post=_np(V_post), v_carry=_np(v_carry),
            ahp=_np(ahp_row, ref=v_carry) if ahp_row is not None else np.zeros((v_carry.shape[-2], v_carry.shape[-1])))
        counter[0] += 1
        return out

    nm.segmented_plif_selective = wrap
    try:
        functional.reset_net(snn)
        with torch.no_grad(), torch.amp.autocast(DEV, dtype=torch.bfloat16):
            _ = model(input_ids=ids)
    finally:
        nm.segmented_plif_selective = orig
    return grabbed


def analyze_layer(d):
    """d: dict from capture. shapes: beta/u/v_th/output/V_post = (TK, b, H); v_init/v_carry/ahp = (b, H); k_t = (T, b)."""
    K = d["K"]
    beta, u, v_th, output, V_post = d["beta"], d["u"], d["v_th"], d["output"], d["V_post"]
    TK, b, H = V_post.shape
    T = TK // K
    # 重建 V_pre[k] = beta[k]·V_post_prev[k] + u[k]，token 内首帧 prev = v_init(扩到所有 token? 不——v_init 只对 token0；
    # token t 的首帧 prev = token t-1 第 k_{t-1} 帧的 V_post). 简化: 用 V_post 直接前移一帧；首帧用 v_init 近似(误差仅 token 边界).
    Vp = V_post.reshape(T, K, b, H)
    bet = beta.reshape(T, K, b, H); uu = u.reshape(T, K, b, H); vth = v_th.reshape(T, K, b, H)
    prev = np.empty_like(Vp)
    prev[:, 1:] = Vp[:, :-1]
    prev[0, 0] = d["v_init"]
    if T > 1:
        prev[1:, 0] = Vp[:-1, K - 1]  # 近似: 用上一 token 末帧(实际应是第 k_{t-1} 帧)
    V_pre = bet * prev + uu  # (T,K,b,H)

    fired = output.reshape(T, K, b, H) != 0.0
    p_fire = float(fired.mean())
    Vpre_q = q(np.abs(V_pre))
    vth_mean = float(vth.mean())

    # retain = V_post / v_th @fire
    if fired.any():
        retain = (Vp[fired] / np.clip(vth[fired], 1e-6, None))
        rq = q(retain)
        frac_05 = float((retain > 0.5).mean())
        frac_10 = float((retain > 1.0).mean())  # 放电后仍超阈 → 膜可单凭余量再次放电
    else:
        rq = (0., 0., 0.); frac_05 = frac_10 = 0.0

    # token 内 K 帧膜存活: |V_post[:,K-1]| / |V_post[:,0]|  (逐 token,b 取中位数, 过滤近零分母)
    a0 = np.abs(Vp[:, 0]).reshape(-1); aK = np.abs(Vp[:, K - 1]).reshape(-1)
    m = a0 > 1e-4
    carry_kframe = float(np.median(aK[m] / a0[m])) if m.any() else float("nan")
    # token→token 携带: |v_carry| / mean |V_post| (token 内)
    carry_tok = float(np.abs(d["v_carry"]).mean() / (np.abs(Vp).mean() + 1e-9))
    return dict(p_fire=p_fire, Vpre_q=Vpre_q, vth=vth_mean, retain_q=rq,
                frac_retain05=frac_05, frac_retain10=frac_10, carry_kframe=carry_kframe, carry_tok=carry_tok)


def main():
    model, cfg = load_model()
    spike_mode = getattr(cfg, "spike_mode", "supra")
    ids = get_val_batch(args.seq, args.batch, cfg.vocab_size)
    g = capture(model, ids, spike_mode)
    print("\n" + "=" * 100)
    print(f"A5 膜电位残留 / K-frame 携带  (spike_mode={spike_mode}, use_ahp={getattr(cfg,'use_ahp',False)}, "
          f"seq={args.seq} batch={args.batch}; SNNBlock.hidden_neuron 逐层)")
    print("=" * 100)
    print(f"{'layer':>6} {'p_fire':>8} {'|V_pre|1/50/90%':>22} {'v_th':>7} {'retain=Vpost/vth@fire 1/50/90':>32} "
          f"{'>0.5':>6} {'>1.0':>6} {'carryKf':>8} {'carry→':>8}")
    rows = []
    for li in sorted(g):
        r = analyze_layer(g[li]); rows.append(r)
        vp = r["Vpre_q"]; rq = r["retain_q"]
        print(f"{li:>6d} {r['p_fire']:>8.3f} {vp[0]:>6.3f}/{vp[1]:>6.3f}/{vp[2]:>6.3f}   {r['vth']:>7.3f} "
              f"{rq[0]:>9.3f}/{rq[1]:>8.3f}/{rq[2]:>8.3f}  {r['frac_retain05']:>6.3f} {r['frac_retain10']:>6.3f} "
              f"{r['carry_kframe']:>8.3f} {r['carry_tok']:>8.3f}")
    # 汇总
    pf = np.mean([r["p_fire"] for r in rows])
    rm = np.mean([r["retain_q"][1] for r in rows])
    f05 = np.mean([r["frac_retain05"] for r in rows]); f10 = np.mean([r["frac_retain10"] for r in rows])
    ck = np.nanmean([r["carry_kframe"] for r in rows]); ct = np.mean([r["carry_tok"] for r in rows])
    print("-" * 100)
    print(f"  均值: p_fire={pf:.3f}  retain(p50)={rm:.3f}  frac(retain>0.5)={f05:.3f}  frac(retain>1.0)={f10:.3f}  "
          f"carry_Kframe={ck:.3f}  carry_token={ct:.3f}")
    if spike_mode == "supra":
        print("  [supra] retain 应恒 ≈1.0 (V_post=min(V_pre,v_th)=v_th @fire)；放电后膜被截到阈值，超阈历史丢弃 → frac(>1.0)≈0.")
    else:
        print("  [quantal] retain = (V_pre-v_th-ahp)/v_th；retain>1.0 的那部分 = 放电后膜仍超阈 → 膜=历史累加器，")
        print("            余量直接进入下一帧/下一 token 的递推 → 这就是 §4b E3 里 quantal 长程影响 ~20×supra 的微观来源。")


if __name__ == "__main__":
    main()
