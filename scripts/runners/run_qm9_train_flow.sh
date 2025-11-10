#!/bin/bash
set -euo pipefail

# Inside-container runner for QM9 latent flow matching training.
# Mirrors run_zinc_train_flow.sh but targets the QM9 unconditional encoder.
# Usage:
#   run_qm9_train_flow.sh \
#       [--checkpoint auto|/path/to/qm9_encoder.ckpt] \
#       [--config cfg/QM9_unconditional_generation_flow.yaml] \
#       [--max-epoch 3000] \
#       [--out-dir /workspace/runs/<exp>] \
#       [--wandb-name name]

checkpoint="auto"
config="cfg/QM9_unconditional_generation_flow.yaml"
max_epoch=3000
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

exp_default="${EXP:-qm9_flow_$(date +%Y%m%d_%H%M%S)}"
out_dir=${out_dir:-"/workspace/runs/$exp_default"}
wandb_name=${wandb_name:-"qm9_flow_${exp_default}"}

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

if [[ "$checkpoint" == "auto" ]]; then
  echo "Auto-detecting latest QM9 unconditional encoder checkpoint..."
  mapfile -t CANDIDATES < <(find /workspace -type f \
    \( -path "*/QM9_unconditional_generation_encoder/*/ckpt/*.ckpt" \
       -o -path "*/qm9_unconditional_generation_encoder/*/ckpt/*.ckpt" \
       -o -path "*/QM9_unconditional*/ckpt/*.ckpt" \) 2>/dev/null | sort -V)
  if (( ${#CANDIDATES[@]} == 0 )); then
    echo "ERROR: No QM9 encoder checkpoint found."
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

echo "Starting QM9 flow matching training..."
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
