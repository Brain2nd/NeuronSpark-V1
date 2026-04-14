"""
NeuronSpark checkpoint → 自包含 HuggingFace artifact 导出。

导出目录包含所有代码文件，支持冷启动:
  AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)

用法:
    python scripts/convert_to_hf.py \
        --checkpoint checkpoints/ckpt_step403000 \
        --tokenizer ./tokenizer/ \
        --output ./neuronspark-hf/
"""

import argparse
import json
import os
import shutil
import sys

import torch
from transformers import AutoTokenizer

# 确保项目根目录在 path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from checkpoint_utils import load_config, load_model_weights


def convert(checkpoint_path, tokenizer_path, output_dir):
    print(f"[1/8] Loading checkpoint config from {checkpoint_path}")
    snn_config = load_config(checkpoint_path)

    print(f"[2/8] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    target_vocab_size = len(tokenizer)
    print(f"       Tokenizer vocab size: {target_vocab_size}")

    # 构建 HF config
    config = NeuronSparkConfig(
        vocab_size=snn_config.get('vocab_size', 64000),
        D=snn_config.get('D', 1024),
        N=snn_config.get('N', 8),
        K=snn_config.get('K', 12),
        num_layers=snn_config.get('num_layers', 24),
        D_ff=snn_config.get('D_ff', 3072),
        v_th_min=snn_config.get('v_th_min', 0.1),
        memory_layer_interval=snn_config.get('memory_layer_interval', 4),
        D_key=snn_config.get('D_key', 128),
        D_value=snn_config.get('D_value', 128),
    )

    print(f"[3/8] Building model (vocab={config.vocab_size})")
    model = NeuronSparkForCausalLM(config)

    print(f"[4/8] Loading weights")
    load_model_weights(checkpoint_path, model.snn, 'cpu')

    # 条件性 vocab 扩容
    current_emb_rows = model.snn.embed_tokens.weight.shape[0]
    if current_emb_rows < target_vocab_size:
        diff = target_vocab_size - current_emb_rows
        print(f"[4b]  Expanding embedding: {current_emb_rows} → {target_vocab_size} (+{diff} rows, 1e-4 init)")
        D = model.snn.embed_tokens.weight.shape[1]
        new_rows = torch.randn(diff, D, dtype=model.snn.embed_tokens.weight.dtype) * 1e-4
        new_weight = torch.cat([model.snn.embed_tokens.weight.data, new_rows], dim=0)
        model.snn.embed_tokens = torch.nn.Embedding.from_pretrained(new_weight, freeze=False)
        # 同步 vocab_size
        model.config.vocab_size = target_vocab_size
        model.snn.vocab_size = target_vocab_size
    elif current_emb_rows > target_vocab_size:
        print(f"  WARNING: embedding ({current_emb_rows}) > tokenizer ({target_vocab_size}), keeping as-is")
    else:
        print(f"  Embedding and tokenizer vocab match: {current_emb_rows}")

    # 保存 HF artifact
    os.makedirs(output_dir, exist_ok=True)

    # 转 bf16 存储（与 Qwen3 等标准模型一致，需要 fp32 的运算在 forward 内部显式上转）
    model = model.to(torch.bfloat16)

    print(f"[5/8] Saving model via save_pretrained → {output_dir}")
    model.save_pretrained(output_dir)

    # config.json dtype 修正为 bfloat16
    import json
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path) as f:
        cfg = json.load(f)
    cfg['dtype'] = 'bfloat16'
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[6/8] Saving tokenizer → {output_dir}")
    tokenizer.save_pretrained(output_dir)

    # 复制代码文件到输出根目录（与 model.safetensors 同级，对齐 HF 标准）
    print(f"[7/8] Copying code files")
    src_pkg = os.path.join(_project_root, 'neuronspark')
    shutil.copy2(os.path.join(src_pkg, 'configuration_neuronspark.py'),
                 os.path.join(output_dir, 'configuration_neuronspark.py'))
    # modeling: 把相对导入改为 importlib 动态加载（绕过 HF check_imports 静态扫描）
    modeling_src = os.path.join(src_pkg, 'modeling_neuronspark.py')
    modeling_dst = os.path.join(output_dir, 'modeling_neuronspark.py')
    with open(modeling_src) as f:
        content = f.read()
    content = content.replace(
        'from .configuration_neuronspark import NeuronSparkConfig',
        'import importlib.util as _ilu\n'
        '_spec = _ilu.spec_from_file_location("configuration_neuronspark", '
        '__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), '
        '"configuration_neuronspark.py"))\n'
        '_mod = _ilu.module_from_spec(_spec)\n'
        '_spec.loader.exec_module(_mod)\n'
        'NeuronSparkConfig = _mod.NeuronSparkConfig',
    )
    with open(modeling_dst, 'w') as f:
        f.write(content)

    print(f"[8/8] Cold-start verification (subprocess, isolated from repo sys.modules)")
    import subprocess
    verify_script = f"""\
import sys, os
# 确保不从仓库目录导入
sys.path = [p for p in sys.path if 'NeuronSpark' not in p]
os.chdir('/tmp')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True).to('cuda:0').eval()
tok = AutoTokenizer.from_pretrained('{output_dir}')
ids = tok('test', return_tensors='pt')['input_ids'].to('cuda:0')
with torch.no_grad():
    out = model(ids)
assert out.logits.shape[-1] == model.config.vocab_size, f"vocab mismatch: {{out.logits.shape[-1]}} vs {{model.config.vocab_size}}"
print(f'COLD_START_OK logits={{out.logits.shape}}')
"""
    result = subprocess.run(
        [sys.executable, '-c', verify_script],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip().splitlines()[-1]}")
        raise RuntimeError(f"Cold-start verification failed:\n{result.stderr}")
    print(f"  {result.stdout.strip()}")

    print(f"\nDone! Self-contained HF artifact at: {output_dir}")
    print(f"Load with:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NeuronSpark checkpoint to self-contained HF artifact')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='NeuronSpark checkpoint 目录 (含 config.json + model.safetensors)')
    parser.add_argument('--tokenizer', type=str, default='./tokenizer/',
                        help='Tokenizer 目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录 (自包含 HF artifact)')
    args = parser.parse_args()

    convert(args.checkpoint, args.tokenizer, args.output)
