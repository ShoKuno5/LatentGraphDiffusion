#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=12:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N lgd_job
#PJM -o lgd_%j.out
#PJM -e lgd_%j.err

# =============================================================
# Edit the variables in this block before submitting with pjsub.
# You can also override them via pjsub -x VAR=... if desired.
# =============================================================

# Dataset and mode
DATASET=${DATASET:-zinc}           # zinc | qm9
MODE=${MODE:-flow}              # encoder | diffusion | flow | uncond

# QM9-specific target property (ignored for ZINC)
TARGET_PROPERTY=${TARGET_PROPERTY:-mu}  # mu|alpha|e_HOMO|e_LUMO|delta_e|cv

# Optional config path (leave empty to use sensible defaults)
CONFIG=${CONFIG:-cfg/my_zinc-flow_rf.yaml}

# Encoder checkpoint (used for diffusion/flow/uncond; ignored for encoder)
CHECKPOINT=${CHECKPOINT:-runs/zinc_encoder_fast_hpc/zinc-encoder-fast/0/ckpt/399.ckpt}     # auto | /path/to/encoder.ckpt

# Training length
REPEAT=${REPEAT:-}                 # encoder/diffusion/uncond only; defaulted below
MAX_EPOCH=${MAX_EPOCH:-20}           # defaulted below

# Optional experiment prefix and WandB naming
EXP_PREFIX=${EXP_PREFIX:-}
WANDB_NAME=${WANDB_NAME:-}
# Optional WandB routing (can also come from .wandbrc or pjsub -x)
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}

# =============================================================

set -euo pipefail

# Remember submission working directory early; PJM may copy scripts to a spool dir
SUBMIT_DIR="${PJM_SUBMIT_DIR:-$PWD}"

source /etc/profile.d/modules.sh || true
if command -v module &>/dev/null; then
  module load singularity/3.7.3 || true
  module load cuda/12.6 || true
fi

# Resolve repo root robustly and source Wisteria environment defaults
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=""
ENV_PATH=""
for cand in \
  "$SUBMIT_DIR" \
  "$SUBMIT_DIR/.." \
  "$SUBMIT_DIR/../LatentGraphDiffusion" \
  "$SCRIPT_DIR" \
  "$SCRIPT_DIR/.." \
  "$SCRIPT_DIR/../.."; do
  if [ -f "$cand/scripts/env/wisteria.sh" ]; then
    REPO_ROOT="$(cd "$cand" && pwd)"; ENV_PATH="$cand/scripts/env/wisteria.sh"; break;
  fi
  if [ -f "$cand/ops/env/wisteria.sh" ]; then
    REPO_ROOT="$(cd "$cand" && pwd)"; ENV_PATH="$cand/ops/env/wisteria.sh"; break;
  fi
done
if [ -z "$ENV_PATH" ]; then
  echo "ERROR: Could not locate env preset (scripts/env/wisteria.sh or ops/env/wisteria.sh) from SUBMIT_DIR=$SUBMIT_DIR or SCRIPT_DIR=$SCRIPT_DIR" >&2
  echo "Hint: submit from the repo root: pjsub scripts/lgd_pjm.sh (or ops/lgd_pjm.sh)" >&2
  exit 1
fi
source "$ENV_PATH"

# Defaults for CONFIG/REPEAT/MAX_EPOCH depending on dataset/mode
if [[ -z "${CONFIG}" ]]; then
  case "${DATASET}" in
    zinc)
      case "${MODE}" in
        encoder) CONFIG="cfg/zinc-encoder.yaml" ;;
        diffusion) CONFIG="cfg/zinc-diffusion_ddpm.yaml" ;;
        flow) CONFIG="cfg/zinc-flow_rf.yaml" ;;
        uncond) CONFIG="cfg/zinc-diffusion_ddpm_unconditional.yaml" ;;
        *) echo "Unsupported MODE for ZINC: ${MODE}"; exit 1;;
      esac
      ;;
    qm9)
      case "${MODE}" in
        encoder) CONFIG="cfg/QM9_regression_encoder_${TARGET_PROPERTY}.yaml" ;;
        diffusion) CONFIG="cfg/QM9-diffusion_ddpm_regression_${TARGET_PROPERTY}.yaml" ;;
        *) echo "Unsupported MODE for QM9: ${MODE} (use encoder or diffusion)"; exit 1;;
      esac
      ;;
    *)
      echo "Unsupported DATASET: ${DATASET}"; exit 1;;
  esac
fi

if [[ -z "${REPEAT}" ]]; then
  case "${DATASET}" in
    qm9) REPEAT=3 ;;
    zinc)
      case "${MODE}" in
        flow) REPEAT=1 ;;  # not used by flow runner; keep placeholder
        *) REPEAT=5 ;;
      esac
      ;;
  esac
fi

if [[ -z "${MAX_EPOCH}" ]]; then
  case "${DATASET}" in
    qm9) MAX_EPOCH=50 ;;
    zinc)
      case "${MODE}" in
        flow) MAX_EPOCH=300 ;;
        *) MAX_EPOCH=50 ;;
      esac
      ;;
  esac
fi

# Experiment naming and directories
TS=$(date +%Y%m%d_%H%M%S)
EXP="${EXP_PREFIX}${DATASET}_${MODE}_${TS}"
EXP_DIR="$RUNS/$EXP"
mkdir -p "$DATA" "$EXP_DIR"

# Configure Weights & Biases (WandB) from .wandbrc if available or env overrides
WBR="$REPO_ROOT/.wandbrc"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WBR" ]; then
  # Shell-safe parse of simple key = value lines
  WANDB_API_KEY="$(awk -F' *= *' '/^api_key/ {print $2}' "$WBR" | head -1 | tr -d '\r\n' )"
  export WANDB_API_KEY
fi
if [ -z "${WANDB_ENTITY:-}" ] && [ -f "$WBR" ]; then
  WANDB_ENTITY="$(awk -F' *= *' '/^entity/ {print $2}' "$WBR" | head -1 | tr -d '\r\n' )"
  export WANDB_ENTITY
fi
if [ -z "${WANDB_PROJECT:-}" ] && [ -f "$WBR" ]; then
  WANDB_PROJECT="$(awk -F' *= *' '/^project/ {print $2}' "$WBR" | head -1 | tr -d '\r\n' )"
  export WANDB_PROJECT
fi
# If we have an API key and WANDB_MODE wasn't explicitly set to offline, go online
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
  export WANDB_MODE=online
fi

# Capture PJM stdout/stderr into run dir in real-time as well
exec > >(tee -a "$EXP_DIR/pjm.stdout") 2> >(tee -a "$EXP_DIR/pjm.stderr" >&2)

# Remember job id to relocate PJM files later
JOBID="${PJM_JOBID:-}"

if [[ -z "${WANDB_NAME}" ]]; then
  WANDB_NAME="${DATASET}_${MODE}_${EXP}"
fi
export WANDB_NAME

echo "=== LGD Wisteria Job ==="
echo "Dataset     : ${DATASET}"
echo "Mode        : ${MODE}"
if [[ "${DATASET}" == "qm9" ]]; then
  echo "QM9 Target  : ${TARGET_PROPERTY}"
fi
echo "Config      : ${CONFIG}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Repeat      : ${REPEAT}"
echo "Max Epoch   : ${MAX_EPOCH}"
echo "Experiment  : ${EXP}"
echo "Out Dir     : ${EXP_DIR}"
echo "W&B Name    : ${WANDB_NAME}"
echo "Start time  : $(date)"

# Select runner base path (supports both scripts/runners and ops/runners layouts)
RUNNER_BASE="/workspace/scripts/runners"
if [ ! -d "$REPO_ROOT/scripts/runners" ] && [ -d "$REPO_ROOT/ops/runners" ]; then
  RUNNER_BASE="/workspace/ops/runners"
fi

# Run inside Singularity and dispatch to the appropriate runner
declare -a BIND_WANDB_ARGS=()
if [ -f "$WBR" ]; then
  BIND_WANDB_ARGS=(-B "$WBR:/workspace/.wandbrc")
fi

singularity exec --nv \
  -B "$CODE":/workspace \
  -B "$RUNS":/workspace/runs \
  -B "$DATA":/workspace/data \
  "${BIND_WANDB_ARGS[@]}" \
  "$IMG" \
  bash -lc "
    set -e;
    cd /workspace;
    export PYTHONPATH=/workspace:\$PYTHONPATH;
    export PYTHONUNBUFFERED=1;
    # Configure WandB from .wandbrc inside container if present
    if [ -f /workspace/.wandbrc ]; then
      export WANDB_API_KEY=\$(awk -F' *= *' '/^api_key/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
      export WANDB_ENTITY=\$(awk -F' *= *' '/^entity/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
      if [ -n "\${WANDB_API_KEY:-}" ] && [ "\${WANDB_MODE:-}" != "offline" ]; then export WANDB_MODE=online; fi;
      echo \"[wandb] entity='\${WANDB_ENTITY:-unset}', mode='\${WANDB_MODE:-unset}'\";
    else
      echo \"[wandb] .wandbrc not found; using inherited env (WANDB_MODE=\${WANDB_MODE:-unset})\";
    fi;
    mkdir -p /workspace/runs/$EXP;

    case '$DATASET' in
      zinc)
        case '$MODE' in
          encoder)
            $RUNNER_BASE/run_zinc_pretrain_encoder.sh \
              --config '$CONFIG' \
              --repeat '$REPEAT' \
              --max-epoch '$MAX_EPOCH' \
              --out-dir '/workspace/runs/$EXP' \
              --wandb-name '$WANDB_NAME';
            ;;
          diffusion)
            $RUNNER_BASE/run_zinc_train_diffusion.sh \
              --checkpoint '$CHECKPOINT' \
              --config '$CONFIG' \
              --repeat '$REPEAT' \
              --max-epoch '$MAX_EPOCH' \
              --out-dir '/workspace/runs/$EXP' \
              --wandb-name '$WANDB_NAME';
            ;;
          flow)
            $RUNNER_BASE/run_zinc_train_flow.sh \
              --checkpoint '$CHECKPOINT' \
              --config '$CONFIG' \
              --max-epoch '$MAX_EPOCH' \
              --out-dir '/workspace/runs/$EXP' \
              --wandb-name '$WANDB_NAME';
            ;;
          uncond)
            $RUNNER_BASE/run_zinc_train_diffusion_uncond.sh \
              --checkpoint '$CHECKPOINT' \
              --config '$CONFIG' \
              --repeat '$REPEAT' \
              --max-epoch '$MAX_EPOCH' \
              --out-dir '/workspace/runs/$EXP' \
              --wandb-name '$WANDB_NAME';
            ;;
          *) echo 'Unsupported MODE for zinc: $MODE'; exit 1;;
        esac
        ;;
      qm9)
        case '$MODE' in
          encoder)
            $RUNNER_BASE/run_qm9_pretrain_encoder.sh \
              --target '$TARGET_PROPERTY' \
              --repeat '$REPEAT' \
              --max-epoch '$MAX_EPOCH' \
              --out-dir '/workspace/runs/$EXP' \
              --wandb-name '$WANDB_NAME';
            ;;
          diffusion)
            $RUNNER_BASE/run_qm9_train_diffusion.sh \
              --target '$TARGET_PROPERTY' \
              --checkpoint '$CHECKPOINT' \
              --repeat '$REPEAT' \
              --max-epoch '$MAX_EPOCH' \
              --out-dir '/workspace/runs/$EXP' \
              --wandb-name '$WANDB_NAME';
            ;;
          *) echo 'Unsupported MODE for qm9: $MODE (use encoder or diffusion)'; exit 1;;
        esac
        ;;
      *) echo 'Unsupported DATASET: $DATASET'; exit 1;;
    esac
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Write a brief completion note
{
  echo "LGD job completed at $(date)";
  echo "Dataset=${DATASET}";
  echo "Mode=${MODE}";
  if [[ "${DATASET}" == "qm9" ]]; then echo "Target=${TARGET_PROPERTY}"; fi
  echo "Config=${CONFIG}";
  echo "Checkpoint=${CHECKPOINT}";
  echo "Repeat=${REPEAT}";
  echo "MaxEpoch=${MAX_EPOCH}";
} > "$EXP_DIR/job_completed.txt"

# List checkpoints if any
find "$EXP_DIR" -name '*.ckpt' -type f -exec ls -la {} + 2>/dev/null || true

# Relocate PJM spool files (lgd_%j.out/err) into the run directory, then symlink back
if [[ -n "$JOBID" ]]; then
  for ext in out err; do
    src="$SUBMIT_DIR/lgd_${JOBID}.$ext"
    dest="$EXP_DIR/lgd_${JOBID}.$ext"
    if [[ -f "$src" ]]; then
      mv -f "$src" "$dest" 2>/dev/null || cp -f "$src" "$dest"
      ln -sf "$dest" "$src" || true
      echo "Moved PJM $ext to: $dest"
    fi
  done
fi
