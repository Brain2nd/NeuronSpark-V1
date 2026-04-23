#!/bin/bash
# H200 SXM full setup: clone repo + install env + download dataset
# Run as root on fresh pod:
#   bash scripts/setup_h200.sh
# Or inline:
#   curl -sSL https://raw.githubusercontent.com/Brain2nd/NeuronSpark-V1/v3/scripts/setup_h200.sh | bash
#
# 显存预算 (bs=4 seq=2048 config C 1.24B @ 8 × H200 141GB):
#   base ~13 GB + 0.0134 × 8192 tok = 122.8 GB/card, 余 18 GB

set -e

REPO_URL="${REPO_URL:-https://github.com/Brain2nd/NeuronSpark-V1.git}"
BRANCH="${BRANCH:-v3}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/NeuronSpark-V1-v3}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data/v3_pretrain_mix}"

echo "============================================"
echo "NeuronSpark v3 H200 setup"
echo "  workspace: $WORKSPACE"
echo "  repo_dir:  $REPO_DIR"
echo "  data_dir:  $DATA_DIR"
echo "============================================"

# --- 1. Redirect caches to data disk (system disk is usually small) ---
echo ""
echo "[1/5] Configuring cache dirs → data disk ..."
mkdir -p "$WORKSPACE/tmp" "$WORKSPACE/torch_cache" "$WORKSPACE/triton_cache" \
         "$WORKSPACE/hf_cache/hub" "$WORKSPACE/pip_cache"
cat >> ~/.bashrc <<EOF

# NeuronSpark v3: data-disk cache redirect (system disk is 20GB, fills fast)
export TMPDIR=$WORKSPACE/tmp
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/torch_cache
export TRITON_CACHE_DIR=$WORKSPACE/triton_cache
export HF_HOME=$WORKSPACE/hf_cache
export HUGGINGFACE_HUB_CACHE=$WORKSPACE/hf_cache/hub
export PIP_CACHE_DIR=$WORKSPACE/pip_cache
EOF
# shellcheck disable=SC1090
source ~/.bashrc

# --- 2. Clone repo ---
echo ""
echo "[2/5] Cloning $BRANCH branch to $REPO_DIR ..."
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR" && git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull
else
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
echo "  → at commit: $(git log --oneline -1)"

# --- 3. Install PyTorch (cu128) + project deps ---
echo ""
echo "[3/5] Installing PyTorch cu128 + deps ..."
# Use --index-url for the PyTorch build, then default index for the rest.
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision torchaudio
pip install -r requirements.txt
pip install git+https://github.com/KellerJordan/Muon

# Sanity: torch + CUDA
python -c "import torch, triton; print(f'torch={torch.__version__}  triton={triton.__version__}  cuda={torch.cuda.is_available()}  ngpus={torch.cuda.device_count()}')"

# --- 4. Download pretrain dataset from HF ---
echo ""
echo "[4/5] Downloading Brain2nd/NeuronSpark-Pretrain-v3 (~38 GB) ..."
mkdir -p "$DATA_DIR"
hf download Brain2nd/NeuronSpark-Pretrain-v3 --repo-type dataset --local-dir "$DATA_DIR"
echo "  → data size: $(du -sh $DATA_DIR | cut -f1)"

# --- 5. Smoke test ---
echo ""
echo "[5/5] 8-GPU smoke test (bs=2 seq=2048, hybrid PLIF, DS ZeRO-2 Adam) ..."
cd "$REPO_DIR"
deepspeed --num_gpus=8 scripts/test_muon_ds.py \
    --deepspeed_config ds_config.json \
    --optimizer adam --zero_stage 2 \
    --config configs/smoke_1p16b.json \
    --batch 2 --seq 2048 \
    2>&1 | tail -5

echo ""
echo "============================================"
echo "Setup complete. To run bs=4 seq=2048 smoke:"
echo "  cd $REPO_DIR"
echo "  deepspeed --num_gpus=8 scripts/test_muon_ds.py \\"
echo "    --deepspeed_config ds_config.json \\"
echo "    --optimizer adam --zero_stage 2 \\"
echo "    --config configs/smoke_1p16b.json \\"
echo "    --batch 4 --seq 2048"
echo "============================================"
