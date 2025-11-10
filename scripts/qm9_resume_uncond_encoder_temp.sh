#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=2:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N lgd_qm9_enc_resume
#PJM -o lgd_resume_%j.out
#PJM -e lgd_resume_%j.err

# =============================================================
# TEMP: Resume QM9 encoder run from an existing runs/<EXP> folder
# Usage example:
#   pjsub -x RESUME_EXP=qm9_uncond_encoder_20251110_141525 \
#         scripts/qm9_resume_uncond_encoder_temp.sh
# Optional overrides:
#   RESUME_RUN_ID=0 (default)   -> sub-folder under EXP (usually seed)
#   WANDB_NAME=...              -> defaults to original EXP name
#   MAX_EPOCH=500               -> forwarded to pretrain.py if needed
# =============================================================

set -euo pipefail

CONFIG=${CONFIG:-cfg/QM9_unconditional_generation_encoder.yaml}
RESUME_EXP=${RESUME_EXP:-qm9_uncond_encoder_20251110_141525}
RESUME_RUN_ID=${RESUME_RUN_ID:-0}
MAX_EPOCH=${MAX_EPOCH:-500}
REPEAT=${REPEAT:-6}
WANDB_NAME=${WANDB_NAME:-$RESUME_EXP}

SUBMIT_DIR="${PJM_SUBMIT_DIR:-$PWD}"

source /etc/profile.d/modules.sh || true
if command -v module &>/dev/null; then
  module load singularity/3.7.3 || true
  module load cuda/12.6 || true
fi

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
  echo "ERROR: Could not locate env preset."
  exit 1
fi
source "$ENV_PATH"

if [ ! -d "$RUNS/$RESUME_EXP" ]; then
  echo "ERROR: Target runs directory '$RUNS/$RESUME_EXP' not found." >&2
  exit 1
fi

RESUME_OUT="$RUNS/$RESUME_EXP"
RESUME_RUN_DIR="$RESUME_OUT/QM9_unconditional_generation_encoder/$RESUME_RUN_ID"
if [ ! -d "$RESUME_RUN_DIR" ]; then
  echo "ERROR: Target run directory '$RESUME_RUN_DIR' not found." >&2
  exit 1
fi

WBR="$REPO_ROOT/.wandbrc"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WBR" ]; then
  WANDB_API_KEY="$(awk -F' *= *' '/^api_key/ {print $2}' "$WBR" | head -1 | tr -d '\r\n')"
  export WANDB_API_KEY
fi
if [ -z "${WANDB_ENTITY:-}" ] && [ -f "$WBR" ]; then
  WANDB_ENTITY="$(awk -F' *= *' '/^entity/ {print $2}' "$WBR" | head -1 | tr -d '\r\n')"
  export WANDB_ENTITY
fi
if [ -z "${WANDB_PROJECT:-}" ] && [ -f "$WBR" ]; then
  WANDB_PROJECT="$(awk -F' *= *' '/^project/ {print $2}' "$WBR" | head -1 | tr -d '\r\n')"
fi
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
  export WANDB_MODE=online
fi

export WANDB_PROJECT="${WANDB_PROJECT:-LGD-QM9-Uncond-Encoder}"
export WANDB_NAME

exec > >(tee -a "$RESUME_OUT/pjm_resume.stdout") 2> >(tee -a "$RESUME_OUT/pjm_resume.stderr" >&2)

JOBID="${PJM_JOBID:-}"

echo "=== Resume QM9 Unconditional Encoder ==="
echo "Config      : ${CONFIG}"
echo "Resume EXP  : ${RESUME_EXP}"
echo "Run ID      : ${RESUME_RUN_ID}"
echo "Out Dir     : ${RESUME_OUT}"
echo "W&B Name    : ${WANDB_NAME}"
echo "Start time  : $(date)"

RUNNER_BASE="/workspace/scripts/runners"
if [ ! -d "$REPO_ROOT/scripts/runners" ] && [ -d "$REPO_ROOT/ops/runners" ]; then
  RUNNER_BASE="/workspace/ops/runners"
fi

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
    set -e
    cd /workspace
    export PYTHONPATH=/workspace:\$PYTHONPATH
    export PYTHONUNBUFFERED=1
    if [ -f /workspace/.wandbrc ]; then
      export WANDB_API_KEY=\$(awk -F' *= *' '/^api_key/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n')
      export WANDB_ENTITY=\$(awk -F' *= *' '/^entity/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n')
      if [ -n \"\${WANDB_API_KEY:-}\" ] && [ \"\${WANDB_MODE:-}\" != \"offline\" ]; then export WANDB_MODE=online; fi
      echo \"[wandb] entity='\${WANDB_ENTITY:-unset}', mode='\${WANDB_MODE:-unset}'\"
    else
      echo \"[wandb] .wandbrc not found; using inherited env (WANDB_MODE=\${WANDB_MODE:-unset})\"
    fi

    python pretrain.py \
      --cfg '$CONFIG' \
      --repeat '$REPEAT' \
      optim.max_epoch '$MAX_EPOCH' \
      train.auto_resume True \
      train.epoch_resume -1 \
      out_dir '/workspace/runs/$RESUME_EXP' \
      wandb.use True \
      wandb.name '$WANDB_NAME'
  "

echo "Resume job completed at: $(date)"
echo "Artifacts left under: $RESUME_OUT"

if [[ -n "$JOBID" ]]; then
  for ext in out err; do
    src="$SUBMIT_DIR/lgd_resume_${JOBID}.$ext"
    dest="$RESUME_OUT/lgd_resume_${JOBID}.$ext"
    if [[ -f "$src" ]]; then
      mv -f "$src" "$dest" 2>/dev/null || cp -f "$src" "$dest"
      ln -sf "$dest" "$src" || true
      echo "Moved PJM $ext to: $dest"
    fi
  done
fi
