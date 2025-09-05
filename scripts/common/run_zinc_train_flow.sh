#!/bin/bash
set -euo pipefail

# Inside-container runner for ZINC flow matching training.
# Usage:
#   run_zinc_train_flow.sh \
#       [--checkpoint auto|/path/to/ckpt.ckpt] \
#       [--config cfg/zinc-flow_rf.yaml] \
#       [--max-epoch 300] \
#       [--out-dir /workspace/runs/<exp>] \
#       [--wandb-name name]

checkpoint="auto"
config="cfg/zinc-flow_rf.yaml"
max_epoch=300
out_dir=""
wandb_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) checkpoint="$2"; shift 2;;
    --config) config="$2"; shift 2;;
    --max-epoch) max_epoch="$2"; shift 2;;
    --out-dir) out_dir="$2"; shift 2;;
    --wandb-name) wandb_name="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Defaults
exp_default="${EXP:-zinc_flow_$(date +%Y%m%d_%H%M%S)}"
out_dir=${out_dir:-"/workspace/runs/$exp_default"}
wandb_name=${wandb_name:-"zinc_flow_${exp_default}"}

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

# Determine checkpoint
if [[ "$checkpoint" == "auto" ]]; then
  echo "Auto-detecting latest ZINC encoder checkpoint..."
  mapfile -t CANDIDATES < <(find /workspace -type f \
    \( -path "*/zinc-encoder/*/ckpt/*.ckpt" \
       -o -path "*/zinc-encoder-fast/*/ckpt/*.ckpt" \
       -o -path "*/zinc_encoder_*/ckpt/*.ckpt" \
       -o -path "*/zinc_encoder_*/*.ckpt" \) 2>/dev/null | sort -V)
  if (( ${#CANDIDATES[@]} == 0 )); then
    echo "ERROR: No encoder checkpoint found."
    find /workspace -name "*.ckpt" | head -20 || true
    exit 1
  fi
  checkpoint="${CANDIDATES[-1]}"
fi

echo "Using encoder checkpoint: $checkpoint"
if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: Checkpoint not found: $checkpoint"
  exit 1
fi

echo "Starting ZINC flow matching training..."
echo "Config: $config"
echo "Max Epoch: $max_epoch"
echo "Out dir: $out_dir"

set +e
python train_diffusion.py \
  --cfg "$config" \
  flow.first_stage_config "$checkpoint" \
  optim.max_epoch "$max_epoch" \
  wandb.use True \
  out_dir "$out_dir" \
  wandb.name "$wandb_name" \
  2>&1 | tee "$out_dir/flow_train.log"
code=$?
set -e

echo "Training finished with exit code: $code"
echo "Checkpoints in out_dir:"
find "$out_dir" -name '*.ckpt' -type f -exec ls -la {} + || true
exit $code

