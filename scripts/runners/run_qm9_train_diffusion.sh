#!/bin/bash
set -euo pipefail

# Inside-container runner for QM9 diffusion training
# Usage:
#   run_qm9_train_diffusion.sh --target <mu|alpha|e_HOMO|e_LUMO|delta_e|cv> \
#       [--checkpoint auto|/path/to/ckpt] [--repeat 3] [--max-epoch 50] \
#       [--out-dir /workspace/runs/<exp>] [--wandb-name name]

target=""
checkpoint="auto"
repeat=3
max_epoch=50
out_dir=""
wandb_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2;;
    --checkpoint) checkpoint="$2"; shift 2;;
    --repeat) repeat="$2"; shift 2;;
    --max-epoch) max_epoch="$2"; shift 2;;
    --out-dir) out_dir="$2"; shift 2;;
    --wandb-name) wandb_name="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$target" ]]; then
  echo "ERROR: --target is required (mu|alpha|e_HOMO|e_LUMO|delta_e|cv)"
  exit 1
fi

config="cfg/QM9-diffusion_ddpm_regression_${target}.yaml"
exp_default="${EXP:-qm9_${target}_diffusion_$(date +%Y%m%d_%H%M%S)}"
out_dir=${out_dir:-"/workspace/runs/$exp_default"}
wandb_name=${wandb_name:-"qm9_${target}_diffusion_${exp_default}"}

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
  echo "Auto-detecting latest QM9 $target encoder checkpoint..."
  mapfile -t CANDIDATES < <(find /workspace -type f \
    -path "*/qm9_${target}_encoder_*/ckpt/*.ckpt" 2>/dev/null | sort -V)
  if (( ${#CANDIDATES[@]} == 0 )); then
    echo "ERROR: No QM9 $target encoder checkpoint found."
    find /workspace -name "*.ckpt" | grep -i qm9 | head -20 || true
    exit 1
  fi
  checkpoint="${CANDIDATES[-1]}"
fi

echo "Using encoder checkpoint: $checkpoint"
if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: Checkpoint not found: $checkpoint"
  exit 1
fi

echo "Starting QM9 $target diffusion training..."
echo "Config: $config, Repeat: $repeat, Max Epoch: $max_epoch"
echo "Out dir: $out_dir"

set +e
python train_diffusion.py \
  --cfg "$config" \
  --repeat "$repeat" \
  wandb.use True \
  optim.max_epoch "$max_epoch" \
  diffusion.first_stage_config "$checkpoint" \
  out_dir "$out_dir" \
  wandb.name "$wandb_name" \
  2>&1 | tee "$out_dir/diffusion_qm9_${target}.log"
code=$?
set -e

echo "Training finished with exit code: $code"
echo "Checkpoints:"
find "$out_dir" -name '*.ckpt' -type f -exec ls -la {} + || true
exit $code

