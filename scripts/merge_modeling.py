"""
将 model.py + atomic_ops/*.py + HF wrapper 合并为单一 modeling_neuronspark.py。

按拓扑序合并，消除内部 import，统一 Triton 初始化，输出到 neuronspark/modeling_neuronspark.py。

用法:
    python scripts/merge_modeling.py
"""

import os
import re
import sys
import textwrap

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 合并输入白名单（按拓扑序）
# ============================================================
SOURCE_FILES = [
    "atomic_ops/snn_base.py",
    "atomic_ops/rms_norm.py",
    "atomic_ops/parallel_scan.py",
    "atomic_ops/plif_node.py",
    "atomic_ops/selective_plif.py",
    "atomic_ops/lateral_inhibition.py",
    "atomic_ops/snn_ffn.py",
    "atomic_ops/snn_block.py",
    "atomic_ops/snn_decoder_layer.py",
    "atomic_ops/snn_attention_decoder_layer.py",
    "model.py",
]

# 需要从 hf_adapter/modeling_neuronspark.py 抽取 NeuronSparkForCausalLM
WRAPPER_FILE = "hf_adapter/modeling_neuronspark.py"

# 内部 import 模式（需要剥离）
INTERNAL_IMPORT_PATTERNS = [
    r"^from \.(snn_base|rms_norm|parallel_scan|plif_node|selective_plif|"
    r"lateral_inhibition|snn_ffn|snn_block|snn_decoder_layer|"
    r"snn_attention_decoder_layer|configuration_neuronspark) import",
    r"^from atomic_ops[\.\w]* import",
    r"^from model import",
    r"^from \.snn_base import",
    r"^from spikingjelly",
    r"^from checkpoint_utils import",
]

# 文件头部 docstring + 外部 import + Triton bootstrap 提取后需要跳过的模式
SKIP_PATTERNS = INTERNAL_IMPORT_PATTERNS + [
    r"^import os$",
    r"^import copy$",
    r"^import math$",
    r"^import torch$",
    r"^import torch\.nn as nn$",
    r"^import torch\.nn\.functional as F$",
    r"^from torch\.utils\.checkpoint import checkpoint$",
    r"^from dataclasses import dataclass$",
    r"^from typing import Optional$",
    r"^from abc import abstractmethod$",
    r"^from spikingjelly",
]

# Triton bootstrap 模式（整块跳过，已在文件头统一声明）
TRITON_BOOTSTRAP_PATTERNS = [
    r"^_SYSTEM_PTXAS\s*=",
    r"^if os\.path\.exists\(_SYSTEM_PTXAS\)",
    r"^\s+os\.environ\[.TRITON_PTXAS_PATH.\]",
    r"^_HAS_TRITON\s*=\s*False",
    r"^try:\s*$",
    r"^\s+import triton$",
    r"^\s+import triton\.language as tl",
    r"^\s+_HAS_TRITON\s*=\s*True",
    r"^except ImportError:",
]


def is_internal_import(line):
    stripped = line.strip()
    for pat in INTERNAL_IMPORT_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False


def is_skip_line(line):
    stripped = line.strip()
    for pat in SKIP_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False


def extract_body(filepath):
    """读取文件，跳过 module docstring、import 行和 Triton 初始化行。返回代码体行。"""
    with open(filepath) as f:
        lines = f.readlines()

    body = []
    in_docstring = False
    docstring_done = False
    skip_triton_block = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 跳过文件头 docstring
        if not docstring_done:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    docstring_done = True
                    i += 1
                    continue
                elif stripped.endswith('"""') and len(stripped) > 3:
                    docstring_done = True
                    i += 1
                    continue
                else:
                    in_docstring = True
                    i += 1
                    continue
            if in_docstring:
                i += 1
                continue

        # 跳过 Triton bootstrap 整块（_SYSTEM_PTXAS 开头到 except ImportError: + pass）
        if re.match(r"^_SYSTEM_PTXAS\s*=", stripped):
            # 跳过整个 Triton init 块直到 "except ImportError:" 后的 "pass"
            while i < len(lines):
                s = lines[i].strip()
                i += 1
                if s == 'pass' and i > 1 and 'except' in lines[i - 2]:
                    break
            # 跳过块后的空行
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            continue

        # 跳过外部 import（已在文件头统一声明）
        if is_skip_line(line):
            i += 1
            continue

        # 避免连续空行
        if stripped == '' and body and body[-1].strip() == '':
            i += 1
            continue

        body.append(line)
        i += 1

    # 去掉头部连续空行
    while body and body[0].strip() == '':
        body.pop(0)

    return body


def extract_wrapper_class(filepath):
    """从 HF wrapper 文件中只提取 NeuronSparkForCausalLM 类定义。
    排除: path hack、from model import、from_snn_checkpoint 方法。
    """
    with open(filepath) as f:
        content = f.read()

    # 找到 class 定义
    match = re.search(r'^class NeuronSparkForCausalLM\(PreTrainedModel\):', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot find NeuronSparkForCausalLM in {filepath}")

    class_start = match.start()
    class_body = content[class_start:]

    # 找到类结束（下一个顶层定义或文件末尾）
    lines = class_body.split('\n')
    class_lines = [lines[0]]
    for line in lines[1:]:
        # 顶层定义（非缩进的 class/def/变量赋值）结束当前类
        if line and not line[0].isspace() and not line.startswith('#'):
            break
        class_lines.append(line)

    # 剥离 from_snn_checkpoint 方法
    result = []
    skip_method = False
    for line in class_lines:
        if '    def from_snn_checkpoint' in line or '    def from_snn_checkpoint' in line.replace('@classmethod', ''):
            skip_method = True
            continue
        if skip_method:
            if line.strip() == '' or (line and line[0] != ' '):
                skip_method = False
            elif line.startswith('    ') and not line.startswith('        '):
                # 新的方法定义
                skip_method = False
            else:
                continue
        if not skip_method:
            result.append(line)

    # 去掉 @classmethod 装饰器（from_snn_checkpoint 前面的）
    final = []
    for i, line in enumerate(result):
        if line.strip() == '@classmethod' and i + 1 < len(result):
            next_line = result[i + 1].strip()
            if 'from_snn_checkpoint' in next_line:
                continue
        final.append(line)

    return '\n'.join(final)


def main():
    output_dir = os.path.join(_project_root, 'neuronspark')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'modeling_neuronspark.py')

    # ============================================================
    # 文件头
    # ============================================================
    header = textwrap.dedent('''\
    """
    NeuronSpark SNN Language Model — 单文件 HuggingFace 兼容实现。

    本文件由 scripts/merge_modeling.py 自动生成，合并自:
      atomic_ops/snn_base.py, rms_norm.py, parallel_scan.py,
      plif_node.py, selective_plif.py, lateral_inhibition.py,
      snn_ffn.py, snn_block.py, snn_decoder_layer.py,
      snn_attention_decoder_layer.py, model.py + HF wrapper

    不要手工编辑此文件。修改请在源文件中进行，然后重新运行合并脚本。
    """

    import copy
    import math
    import os
    from dataclasses import dataclass
    from typing import Optional

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from .configuration_neuronspark import NeuronSparkConfig

    # Triton (可选 GPU 加速)
    _SYSTEM_PTXAS = '/usr/local/cuda-13.0/bin/ptxas'
    if os.path.exists(_SYSTEM_PTXAS) and 'TRITON_PTXAS_PATH' not in os.environ:
        os.environ['TRITON_PTXAS_PATH'] = _SYSTEM_PTXAS

    _HAS_TRITON = False
    try:
        import triton
        import triton.language as tl
        _HAS_TRITON = True
    except ImportError:
        pass


    ''')

    # ============================================================
    # 按拓扑序合并各 section
    # ============================================================
    sections = []
    for relpath in SOURCE_FILES:
        filepath = os.path.join(_project_root, relpath)
        section_name = os.path.splitext(os.path.basename(relpath))[0]
        body_lines = extract_body(filepath)
        body = ''.join(body_lines).rstrip() + '\n'
        section = f'\n# {"=" * 60}\n# Section: {section_name}\n# {"=" * 60}\n\n{body}\n'
        sections.append(section)

    # HF wrapper section — 固定模板
    wrapper_class = textwrap.dedent('''\
    class NeuronSparkForCausalLM(PreTrainedModel):
        config_class = NeuronSparkConfig
        supports_gradient_checkpointing = False
        _tied_weights_keys = []
        all_tied_weights_keys = {}

        def __init__(self, config: NeuronSparkConfig):
            super().__init__(config)
            self.snn = SNNLanguageModel(
                vocab_size=config.vocab_size,
                D=config.D,
                N=config.N,
                K=config.K,
                num_layers=config.num_layers,
                D_ff=config.D_ff,
                v_th_min=config.v_th_min,
                memory_layer_interval=config.memory_layer_interval,
                D_key=config.D_key,
                D_value=config.D_value,
            )

        def get_input_embeddings(self):
            return self.snn.embed_tokens

        def set_input_embeddings(self, value):
            self.snn.embed_tokens = value

        def get_output_embeddings(self):
            return None

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            out = self.snn(input_ids)
            logits = out.logits
            loss = None
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                if attention_mask is not None:
                    shift_mask = attention_mask[:, 1:].contiguous()
                    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1), ignore_index=-100,
                )
            return CausalLMOutputWithPast(loss=loss, logits=logits)

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

        def can_generate(self):
            return True

        _SENTINEL = object()

        @torch.no_grad()
        def generate(self, input_ids=None, max_new_tokens=_SENTINEL,
                     temperature=_SENTINEL, top_k=_SENTINEL, top_p=_SENTINEL,
                     repetition_penalty=_SENTINEL, eos_token_id=_SENTINEL, **kwargs):
            defaults = dict(max_new_tokens=256, temperature=1.0, top_k=50,
                            top_p=1.0, repetition_penalty=1.0, eos_token_id=None)
            gen_config = kwargs.get('generation_config', None)
            if gen_config is not None:
                for key in defaults:
                    v = getattr(gen_config, key, None)
                    if v is not None:
                        defaults[key] = v
                if not getattr(gen_config, 'do_sample', True):
                    defaults['temperature'] = 0.0
            S = self._SENTINEL
            if max_new_tokens is not S: defaults['max_new_tokens'] = max_new_tokens
            if temperature is not S: defaults['temperature'] = temperature
            if top_k is not S: defaults['top_k'] = top_k
            if top_p is not S: defaults['top_p'] = top_p
            if repetition_penalty is not S: defaults['repetition_penalty'] = repetition_penalty
            if eos_token_id is not S: defaults['eos_token_id'] = eos_token_id
            if not kwargs.get('do_sample', True):
                defaults['temperature'] = 0.0
            if 'max_length' in kwargs and input_ids is not None:
                derived = kwargs['max_length'] - input_ids.shape[1]
                if derived <= 0:
                    return input_ids
                defaults['max_new_tokens'] = derived
            if kwargs.get('num_beams', 1) != 1:
                raise NotImplementedError("NeuronSpark SNN does not support beam search")
            if kwargs.get('num_return_sequences', 1) != 1:
                raise NotImplementedError("NeuronSpark SNN does not support multiple return sequences")
            if 'attention_mask' in kwargs:
                mask = kwargs['attention_mask']
                if mask is not None and mask.min() == 0:
                    raise ValueError("NeuronSpark SNN generate does not support padding in attention_mask")
            if defaults['eos_token_id'] is None:
                defaults['eos_token_id'] = self.config.eos_token_id
            return self.snn.generate(
                    input_ids, max_new_tokens=defaults['max_new_tokens'],
                    temperature=defaults['temperature'], top_k=defaults['top_k'],
                    top_p=defaults['top_p'], repetition_penalty=defaults['repetition_penalty'],
                    eos_token_id=defaults['eos_token_id'],
                )
    ''')

    wrapper_section = f'\n# {"=" * 60}\n# Section: HF Wrapper\n# {"=" * 60}\n\n{wrapper_class}\n'
    sections.append(wrapper_section)

    # ============================================================
    # 写入
    # ============================================================
    with open(output_file, 'w') as f:
        f.write(header)
        for section in sections:
            f.write(section)

    # 验证语法
    import py_compile
    try:
        py_compile.compile(output_file, doraise=True)
        print(f"✓ Generated: {output_file}")
        total_lines = sum(1 for _ in open(output_file))
        print(f"  Total lines: {total_lines}")
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error in generated file: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
