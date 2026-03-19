"""找到保存 checkpoint 时 _neuron_keys set 的遍历顺序，修复 optimizer state。"""
import torch
import sys
import itertools
sys.path.insert(0, ".")
from model import SNNLanguageModel

# 加载 checkpoint optimizer state
ts = torch.load("checkpoints/ckpt_step9000/training_state.pth", map_location="cpu", weights_only=False)
saved_state = ts["optimizer_state"]
saved_pgs = saved_state["param_groups"]
saved_st = saved_state["state"]

# 创建模型，获取 param groups
m = SNNLanguageModel(vocab_size=64000, D=1024, N=8, K=12, num_layers=20, D_ff=3072).bfloat16()
_pg = m.get_param_groups()
_neuron_keys_list = ["input_neurons", "b_beta", "b_alpha", "b_th",
                     "block_output_neuron", "ffn_neurons", "output_neuron"]

# group1 (neuron) 的 saved state shapes
g1_indices = saved_pgs[1]["params"]
saved_neuron_shapes = []
for idx in g1_indices:
    saved_neuron_shapes.append(tuple(saved_st[idx]["exp_avg"].shape))

print(f"Saved neuron param count: {len(saved_neuron_shapes)}")
print(f"Saved neuron shapes (first 10): {saved_neuron_shapes[:10]}")

# other group 也要检查
g0_indices = saved_pgs[0]["params"]
saved_other_shapes = [tuple(saved_st[idx]["exp_avg"].shape) for idx in g0_indices]

# 同样获取 other_params 的 shapes（dict items 顺序是固定的）
# other_params 的构造: [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]
# dict.items() 顺序固定，只要 _neuron_keys 内容不变（membership test），other 顺序就不变
other_params_check = [p for k, ps in _pg.items() if k not in set(_neuron_keys_list) for p in ps]
other_shapes = [tuple(p.shape) for p in other_params_check]
print(f"\nOther param count: saved={len(saved_other_shapes)}, current={len(other_shapes)}")
print(f"Other shapes match: {other_shapes == saved_other_shapes}")

# 暴力搜索 neuron_keys 排列
found = False
for perm in itertools.permutations(_neuron_keys_list):
    neuron_params = [p for k in perm for p in _pg[k]]
    current_shapes = [tuple(p.shape) for p in neuron_params]
    if current_shapes == saved_neuron_shapes:
        print(f"\nFOUND: {perm}")
        found = True

        # 用找到的 perm 顺序加载 state 到临时 optimizer（验证可加载）
        from torch.optim import Adam
        all_other = [p for k, ps in _pg.items() if k not in set(_neuron_keys_list) for p in ps]
        all_neuron_old_order = neuron_params  # perm 顺序
        opt_old = Adam([
            {"params": all_other, "lr": 2e-4, "lr_mult": 1.0},
            {"params": all_neuron_old_order, "lr": 2e-3, "lr_mult": 10.0},
        ])
        opt_old.load_state_dict(saved_state)
        print("Optimizer state loaded with original order!")

        # 重新用 tuple 固定顺序创建 optimizer，逐参数迁移 state
        _neuron_keys_tuple = ("input_neurons", "b_beta", "b_alpha", "b_th",
                              "block_output_neuron", "ffn_neurons", "output_neuron")
        all_neuron_new_order = [p for k in _neuron_keys_tuple for p in _pg[k]]
        opt_new = Adam([
            {"params": all_other, "lr": 2e-4, "lr_mult": 1.0},
            {"params": all_neuron_new_order, "lr": 2e-3, "lr_mult": 10.0},
        ])
        # 做一步 dummy 让 opt_new 初始化 state 结构
        for p in list(all_other) + list(all_neuron_new_order):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        opt_new.step()
        opt_new.zero_grad()

        # 用 param data_ptr 做映射：old optimizer 的 state → new optimizer
        old_ptr_to_state = {}
        for pg in opt_old.param_groups:
            for p in pg["params"]:
                if p.data_ptr() in dict(opt_old.state):
                    pass
                st = opt_old.state.get(p, None)
                if st is not None:
                    old_ptr_to_state[p.data_ptr()] = st

        for pg in opt_new.param_groups:
            for p in pg["params"]:
                old_st = old_ptr_to_state.get(p.data_ptr())
                if old_st is not None:
                    opt_new.state[p] = old_st

        # 保存修复后的 training_state（tuple 顺序）
        ts["optimizer_state"] = opt_new.state_dict()
        torch.save(ts, "checkpoints/ckpt_step9000/training_state.pth")
        print("Fixed training_state.pth saved (tuple order)!")
        break

if not found:
    print("\nNO matching permutation found")
