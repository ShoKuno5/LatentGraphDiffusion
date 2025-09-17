#!/bin/bash
set -euo pipefail

# Inside-container runner for ZINC encoder pretraining
# Usage:
#   run_zinc_pretrain_encoder.sh [--config cfg/zinc-encoder.yaml] [--repeat 5] [--max-epoch 50] [--out-dir /workspace/runs/<exp>] [--wandb-name name]

config="cfg/zinc-encoder.yaml"
repeat=5
max_epoch=50
out_dir=""
wandb_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="$2"; shift 2;;
    --repeat) repeat="$2"; shift 2;;
    --max-epoch) max_epoch="$2"; shift 2;;
    --out-dir) out_dir="$2"; shift 2;;
    --wandb-name) wandb_name="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

exp_default="${EXP:-zinc_encoder_$(date +%Y%m%d_%H%M%S)}"
out_dir=${out_dir:-"/workspace/runs/$exp_default"}
wandb_name=${wandb_name:-"zinc_encoder_${exp_default}"}

mkdir -p "$out_dir"

echo "Env check:"
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
try:
    import torch_geometric
    print(f"PyG: {torch_geometric.__version__}")
except Exception as e:
    print(f"PyG import failed: {e}")
PY

echo "Starting ZINC encoder pretraining..."
echo "Config: $config, Repeat: $repeat, Max Epoch: $max_epoch"
echo "Out dir: $out_dir"

set +e
python pretrain.py \
  --cfg "$config" \
  --repeat "$repeat" \
  wandb.use True \
  optim.max_epoch "$max_epoch" \
  out_dir "$out_dir" \
  wandb.name "$wandb_name" \
  2>&1 | tee "$out_dir/pretrain_full.log"
code=$?
set -e

echo "Pretraining finished with exit code: $code"
echo "Checkpoints:"
find "$out_dir" -name '*.ckpt' -type f -exec ls -la {} + || true
exit $code

