"""
权重适配脚本：将原始 V_post 训练的权重适配到泄漏量激活代码。

源码已使用 (1-β)·V_post 作为 PLIFNode 激活值，
下游投影权重需要列缩放 ÷(1-β) 补偿。

仅修改下游投影权重，不碰 RMSNorm/v_th/神经元参数。

用法:
    python convert_to_leakage.py \
        --input checkpoints_sft/ckpt_step6500.pth \
        --output checkpoints_sft/ckpt_step6500_adapted.pth
"""

import argparse
import torch
from model import SNNLanguageModel


def adapt_weights(model: SNNLanguageModel):
    """适配下游投影权重以匹配泄漏量激活代码。"""
    with torch.no_grad():
        for layer in model.layers:
            # input_neuron1 → SNNBlock 六条投影
            beta1 = torch.sigmoid(layer.input_neuron1.w)
            inv_leak1 = 1.0 / (1.0 - beta1)  # (D,)

            block = layer.snn_block
            for W in [block.W_in, block.W_beta_x, block.W_alpha_x,
                       block.W_th_x, block.W_gate, block.W_skip]:
                W.weight.data.mul_(inv_leak1.unsqueeze(0))

            # input_neuron2 → SNNFFN 三条投影
            beta2 = torch.sigmoid(layer.input_neuron2.w)
            inv_leak2 = 1.0 / (1.0 - beta2)

            ffn = layer.snn_ffn
            for W in [ffn.gate_proj, ffn.up_proj, ffn.skip_proj]:
                W.weight.data.mul_(inv_leak2.unsqueeze(0))

            # gate/up neuron → down_proj
            beta_g = torch.sigmoid(ffn.gate_neuron.w)
            beta_u = torch.sigmoid(ffn.up_neuron.w)
            inv_combined = 1.0 / ((1.0 - beta_g) * (1.0 - beta_u))
            ffn.down_proj.weight.data.mul_(inv_combined.unsqueeze(0))

        # output_neuron → decode_proj
        beta_out = torch.sigmoid(model.output_neuron.w)
        inv_leak_out = 1.0 / (1.0 - beta_out)
        model.decode_proj.weight.data.mul_(inv_leak_out.unsqueeze(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    config = ckpt.get('model_config', {})

    model = SNNLanguageModel(
        vocab_size=config.get('vocab_size', 6144),
        D=config.get('D', 1024), N=config.get('N', 8), K=config.get('K', 32),
        num_layers=config.get('num_layers', 20), D_ff=config.get('D_ff', 3072),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    adapt_weights(model)

    ckpt['model_state_dict'] = model.state_dict()
    torch.save(ckpt, args.output)
    print(f"Saved: {args.output}")
