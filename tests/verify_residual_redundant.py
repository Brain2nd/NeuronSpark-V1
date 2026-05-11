"""实测验证: 残差流 h 在 K 维上是否逐 bit 冗余 (每 token 的 K 份完全相同).

在每一层 forward 后捕获 h, 检查 h.view(seq, K, b, D) 沿 K 轴的最大偏差.
如果全程 == 0 (或机器精度 0), 则简化 h 到 (T,B,D) 不产生计算偏差.

测两种 dtype: fp32 (严格 bit-exact) 和 bf16 (实际训练精度).
"""
import sys, torch
sys.path.insert(0, "/home/dgxspark/Desktop/NeuronSpark-V1")
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM


def check_redundancy(model, ids, K, label):
    """Hook 每层输出, 检查 K 轴偏差."""
    devs = []
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            # out 可能是 (h, ponder_cost) tuple
            h = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(h):
                return
            TK = h.shape[0]
            if TK % K != 0:
                return
            seq = TK // K
            hv = h.view(seq, K, *h.shape[1:])
            dev = (hv - hv[:, 0:1]).abs().max().item()
            devs.append((name, dev))
        return hook

    for i, layer in enumerate(model.snn.layers):
        h = layer.register_forward_hook(make_hook(f"layer{i}({type(layer).__name__})"))
        handles.append(h)

    with torch.no_grad():
        # 直接调 snn 拿到 layer 间的 h
        _ = model.snn(ids)

    for h in handles:
        h.remove()

    print(f"\n=== {label} ===")
    max_dev = 0.0
    for name, dev in devs:
        flag = "OK" if dev == 0.0 else ("~0" if dev < 1e-5 else "!!!")
        print(f"  {name:35s} K-axis max dev = {dev:.3e}  [{flag}]")
        max_dev = max(max_dev, dev)
    print(f"  >>> overall max K-axis deviation: {max_dev:.3e}")
    return max_dev


def main():
    torch.manual_seed(0)
    K = 6
    cfg = NeuronSparkConfig(vocab_size=512, D=128, N=4, K=K, num_layers=8, D_ff=256)

    for dtype, label in [(torch.float32, "fp32 (strict bit-exact)"), (torch.bfloat16, "bf16 (training precision)")]:
        torch.manual_seed(0)
        model = NeuronSparkForCausalLM(cfg)
        # 把 neuron 参数保 fp32, 矩阵转目标 dtype (对齐 from_pretrained 混合精度)
        for n, p in model.named_parameters():
            if n.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th')):
                p.data = p.data.float()
            else:
                p.data = p.data.to(dtype if dtype != torch.float32 else torch.float32)
        for _, b in model.named_buffers():
            if b.is_floating_point():
                b.data = b.data.to(dtype if dtype != torch.float32 else torch.float32)
        model = model.cuda().eval()
        ids = torch.randint(0, 512, (2, 16)).cuda()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16) if dtype == torch.bfloat16 else torch.no_grad():
            md = check_redundancy(model, ids, K, label)
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
        if md == 0.0:
            print(f"  ==> {label}: 严格逐 bit 相同, 简化 h 到 (T,B,D) 零偏差 ✓")
        elif md < 1e-4:
            print(f"  ==> {label}: 偏差 {md:.1e} (机器精度量级), 简化基本无影响")
        else:
            print(f"  ==> {label}: ⚠️ 偏差 {md:.1e} —— 残差流 K 维不是纯冗余!")


if __name__ == "__main__":
    main()
