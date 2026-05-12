"""检查 NeuronSpark V4 模型的混合精度 (矩阵 bf16 / 逐通道神经元参数 .w/.v_th/.ahp (1D tensor) fp32) +
量一下推理常驻 state 大小 (与上下文长度无关). FPGA sizing 用.

用法: CUDA_VISIBLE_DEVICES=N python scripts/v4_check_dtype.py [--D 1024 --N 8 --K 12 --num_layers 24 --D_ff 3072 --vocab 128387]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from neuronspark.modeling_neuronspark import functional

ap = argparse.ArgumentParser()
ap.add_argument("--D", type=int, default=1024)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--K", type=int, default=12)
ap.add_argument("--num_layers", type=int, default=24)
ap.add_argument("--D_ff", type=int, default=3072)
ap.add_argument("--memory_layer_interval", type=int, default=4)
ap.add_argument("--vocab", type=int, default=128387)
ap.add_argument("--spike_mode", default="quantal")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = NeuronSparkConfig(vocab_size=args.vocab, D=args.D, N=args.N, K=args.K, num_layers=args.num_layers,
                        D_ff=args.D_ff, memory_layer_interval=args.memory_layer_interval,
                        spike_mode=args.spike_mode, use_ahp=False)
m = NeuronSparkForCausalLM(cfg).to(dev)

# apply mixed precision: 同 utils/param_groups.promote_neuron_params_fp32 + 矩阵 bf16
for n, p in m.named_parameters():
    p.data = p.data.to(torch.bfloat16)  # 全 bf16 (含神经元参数 .w/.v_th/.ahp —— MAL stochastic rounding); k_predictor EMA buffer 仍 fp32

byd = {}
for n, p in m.named_parameters():
    byd.setdefault(p.dtype, [0, 0]); byd[p.dtype][0] += 1; byd[p.dtype][1] += p.numel()
tot_params = sum(p.numel() for p in m.parameters())
tot_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
print(f"=== config: D={args.D} N={args.N} K={args.K} layers={args.num_layers} D_ff={args.D_ff} vocab={args.vocab} ===")
print("=== param dtype breakdown ===")
for d, (cnt, nel) in byd.items():
    print(f"  {str(d):20s}: {cnt:4d} tensors, {nel/1e6:9.3f}M params, {nel*(2 if d==torch.bfloat16 else 4)/1e6:9.1f} MB")
print(f"total: {tot_params/1e9:.3f}B params, {tot_bytes/1e9:.4f} GB (current mixed precision)")
print(f"  → if all bf16:  {tot_params*2/1e9:.4f} GB")
print(f"  → if all int8:  {tot_params*1/1e9:.4f} GB")
print(f"  → if all int4:  {tot_params*0.5/1e9:.4f} GB")

print("=== spot-check dtypes ===")
pd = dict(m.named_parameters())
for n in ['snn.embed_tokens.weight', 'snn.layers.0.snn_block.W_in.weight', 'snn.layers.0.snn_block.W_beta_x.weight',
          'snn.layers.0.input_neuron1.w', 'snn.layers.0.input_neuron1.v_th', 'snn.norm.gain', 'snn.decode_proj.weight']:
    if n in pd: print(f"  {n:55s} {pd[n].dtype}  (expect: 全 bf16, 含 .w/.v_th)")
print("b_* params (expect []):", [n for n, _ in m.named_parameters() if n.endswith((".b_beta", ".b_alpha", ".b_th"))])
print("Linear bias=True (expect [] aside from k_predictor buffers):",
      [n for n, _ in m.named_parameters() if n.endswith(".bias") and "k_predictor" not in n])

# === 推理常驻 state (与上下文长度无关) ===
m.eval()
functional.reset_net(m.snn)
with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16):
    _ = m(input_ids=torch.randint(0, args.vocab, (1, 8)).to(dev))
state_bytes = 0; state_items = []
for name, mod in m.named_modules():
    for attr in ('v', 'M_state'):
        v = getattr(mod, attr, None)
        if isinstance(v, torch.Tensor) and v.numel():
            state_bytes += v.numel() * v.element_size()
            state_items.append((name, attr, tuple(v.shape), str(v.dtype)))
print(f"=== carried state (batch=1, after one forward) ≈ {state_bytes/1e6:.3f} MB  —— 不随上下文长度增长 (无 KV cache) ===")
for name, attr, shp, dt in state_items[:8]:
    print(f"  {name}.{attr}: {shp} {dt}")
print(f"  ... ({len(state_items)} state tensors total)")
