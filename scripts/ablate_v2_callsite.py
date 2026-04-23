"""逐 PLIF 调用点切换 V1/V2，看哪个调用点造成集成测试 18% 慢速。

V2 在 isolated kernel 里反而略快（-4%），但集成测试慢 18%。
矛盾 → 慢不在 kernel 里。假设：在某个 call-site 的 `.detach()` / view 行为上。

策略：对每个 call site，独立把 V2 切回 V1，其他保持 V2。看哪个切回能恢复速度。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from neuronspark import modeling_neuronspark as mns
from utils.param_groups import promote_neuron_params_fp32


def _shim_rowparam_as_v1():
    """让 plif_rowparam_forward_v2 走 V1 kernel（返回 V_post[-1] 作为 v_last）."""
    _orig_v1 = mns.plif_rowparam_forward
    def _v1(beta_row, u, v_th_row, v_init):
        output, V_post = _orig_v1(beta_row, u, v_th_row, v_init)
        return output, V_post[-1]  # view
    mns.plif_rowparam_forward_v2 = _v1


def _shim_parallel_as_v1():
    """让 plif_parallel_forward_v2 走 V1 kernel."""
    _orig_v1 = mns.plif_parallel_forward
    def _v1(beta, u, v_th, v_init):
        output, V_post, _ = _orig_v1(beta, u, v_th, v_init)
        return output, V_post[-1]
    mns.plif_parallel_forward_v2 = _v1


def _restore():
    """Reload module to restore originals — actually simpler: just re-import."""
    import importlib
    importlib.reload(mns)


def run(mode: str):
    # Re-import to clean state
    import importlib
    importlib.reload(mns)
    if mode == 'v1_both':
        _shim_rowparam_as_v1()
        _shim_parallel_as_v1()
    elif mode == 'v1_rowparam_only':
        _shim_rowparam_as_v1()
    elif mode == 'v1_parallel_only':
        _shim_parallel_as_v1()
    elif mode == 'v2_both':
        pass  # default
    else:
        raise ValueError(mode)

    cfg = NeuronSparkConfig(
        D=512, N=16, K=12, num_layers=6, D_ff=1536,
        vocab_size=1024, memory_layer_interval=4,
    )
    m = mns.NeuronSparkForCausalLM(cfg).cuda().to(torch.bfloat16)
    promote_neuron_params_fp32(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        x = torch.randint(0, cfg.vocab_size, (1, 2048), device='cuda')
        y = torch.randint(0, cfg.vocab_size, (1, 2048), device='cuda')
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = m.snn(x, y)
            loss = out.last_loss.mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # Timed
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(5):
        x = torch.randint(0, cfg.vocab_size, (1, 2048), device='cuda')
        y = torch.randint(0, cfg.vocab_size, (1, 2048), device='cuda')
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = m.snn(x, y)
            loss = out.last_loss.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    del m, opt, out, loss, x, y
    torch.cuda.empty_cache()

    return elapsed / 5, peak


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required"); sys.exit(1)
    print("Config: D=512, N=16, K=12, layers=6, seq=2048, bs=1\n")
    print(f"{'mode':<25s} {'time ms':>10s} {'peak GB':>10s}")
    print("-" * 48)
    for mode in ['v2_both', 'v1_rowparam_only', 'v1_parallel_only', 'v1_both']:
        t, p = run(mode)
        print(f"{mode:<25s} {t*1000:>10.1f} {p:>10.2f}")
