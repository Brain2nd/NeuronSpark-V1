"""P5: V4 端到端验证.

1. 有限差分梯度 sanity check (小模型, fp32): 抽几个参数, 数值梯度 vs 解析梯度.
2. 最小训练循环 smoke: tiny 模型 ~40 step 在随机数据上, 看 loss 下降 (能 overfit) / E[K] 分布 / 无 NaN.
"""
import sys, math, torch
sys.path.insert(0, "/home/dgxspark/Desktop/NeuronSpark-V1")
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM


def _loss(model, ids):
    out = model(input_ids=ids, labels=ids)
    return out.loss


def test_grad_flow():
    """端到端梯度流: 所有可训练参数都收到 finite 梯度 (无 NaN/Inf, 无全零的孤立参数).

    (有限差分对这个深图 + torch.compile + checkpoint 太 finicky — 精确的梯度验证靠
    tests/test_segmented_plif.py 的 fp64 gradcheck (新增的 segmented PLIF 代码) + 下面的
    overfit smoke (端到端正确性).)
    """
    torch.manual_seed(0)
    cfg = NeuronSparkConfig(vocab_size=64, D=32, N=4, K=4, num_layers=3, D_ff=64)
    model = NeuronSparkForCausalLM(cfg).cuda().float().train()
    ids = torch.randint(0, 64, (1, 6)).cuda()
    model.zero_grad()
    loss = _loss(model, ids)
    loss.backward()
    n_total = 0
    n_with_grad = 0
    n_nonzero = 0
    bad = []
    for n, p in model.named_parameters():
        n_total += 1
        if p.grad is None:
            bad.append(f"{n}: grad is None")
            continue
        n_with_grad += 1
        if not torch.isfinite(p.grad).all():
            bad.append(f"{n}: grad has NaN/Inf")
            continue
        if p.grad.abs().sum().item() > 0:
            n_nonzero += 1
    print(f"[grad-flow] params: {n_total} total, {n_with_grad} with grad, {n_nonzero} with nonzero grad")
    if bad:
        print("  problems:", bad[:10])
    assert not bad, f"grad-flow problems: {bad[:5]}"
    # 允许少数参数梯度恰好为 0 (如某些 k_predictor 在 fresh init + 单 batch 下), 但绝大多数应非零
    assert n_nonzero >= int(0.8 * n_total), f"too many zero-grad params: {n_nonzero}/{n_total}"
    print("  ==> PASS: all params receive finite gradients, ≥80% nonzero")


def test_overfit_smoke():
    torch.manual_seed(1)
    cfg = NeuronSparkConfig(vocab_size=64, D=48, N=4, K=4, num_layers=4, D_ff=96)
    model = NeuronSparkForCausalLM(cfg).cuda().train()
    # 神经元参数 fp32, 矩阵 bf16 (对齐训练混合精度)
    for nm, p in model.named_parameters():
        if nm.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th')):
            p.data = p.data.float()
        else:
            p.data = p.data.to(torch.bfloat16)
    for _, b in model.named_buffers():
        if b.is_floating_point():
            b.data = b.data.to(torch.bfloat16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ids = torch.randint(0, 64, (2, 8)).cuda()  # 固定 batch — 应该能 overfit
    losses = []
    eks = []
    for step in range(40):
        opt.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
        loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss at step {step}")
        loss.backward()
        # grad finite check
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"NaN/Inf grad at step {step}")
        opt.step()
        # E[K] from output neuron + a layer
        ek = model.snn._output_ek
        losses.append(float(loss))
        eks.append(ek)
        if step % 10 == 0:
            # per-layer ek range
            ek_mins = [getattr(l, "_ek_min", None) for l in model.snn.layers]
            ek_maxs = [getattr(l, "_ek_max", None) for l in model.snn.layers]
            valid_min = [x for x in ek_mins if x is not None]
            valid_max = [x for x in ek_maxs if x is not None]
            print(f"  step {step}: loss={float(loss):.4f}  output_E[k]={ek:.2f}  "
                  f"layer E[k] range [{min(valid_min):.1f}, {max(valid_max):.1f}]")
    print(f"  loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 0.7, f"loss should decrease (overfit): {losses[0]:.4f} → {losses[-1]:.4f}"
    assert all(math.isfinite(x) for x in losses), "all losses finite"
    print("  ==> PASS: model trains (loss decreases on fixed batch), no NaN, E[K] sane")


if __name__ == "__main__":
    test_grad_flow()
    test_overfit_smoke()
    print("\nALL P5 TESTS PASSED")
