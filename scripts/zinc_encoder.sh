#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=12:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N lgd_encoder
#PJM -o lgd_%j.out
#PJM -e lgd_%j.err

# Default overrides (set via pjsub -x VAR=value)
CONFIG=${CONFIG:-cfg/my_zinc-encoder_baseline.yaml}
MAX_EPOCH=${MAX_EPOCH:-50}
EXP_PREFIX=${EXP_PREFIX:-}
WANDB_NAME=${WANDB_NAME:-}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}

set -euo pipefail

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
  echo "ERROR: Could not locate env preset for Wisteria." >&2
  exit 1
fi
source "$ENV_PATH"

TS=$(date +%Y%m%d_%H%M%S)
EXP="${EXP_PREFIX}zinc_encoder_${TS}"
EXP_DIR="$RUNS/$EXP"
mkdir -p "$DATA" "$EXP_DIR"

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
  export WANDB_PROJECT
fi
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
  export WANDB_MODE=online
fi

exec > >(tee -a "$EXP_DIR/pjm.stdout") 2> >(tee -a "$EXP_DIR/pjm.stderr" >&2)
JOBID="${PJM_JOBID:-}"
if [[ -z "${WANDB_NAME}" ]]; then
  WANDB_NAME="${EXP}"
fi
export WANDB_NAME

echo "=== LGD Encoder Job ==="
echo "Config      : ${CONFIG}"
echo "Max Epoch   : ${MAX_EPOCH}"
echo "Experiment  : ${EXP}"
echo "Out Dir     : ${EXP_DIR}"
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
    set -e;
    cd /workspace;
    export PYTHONPATH=/workspace:\$PYTHONPATH;
    export PYTHONUNBUFFERED=1;
    if [ -f /workspace/.wandbrc ]; then
      export WANDB_API_KEY=\$(awk -F' *= *' '/^api_key/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
      export WANDB_ENTITY=\$(awk -F' *= *' '/^entity/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
      if [ -n "\${WANDB_API_KEY:-}" ] && [ "\${WANDB_MODE:-}" != "offline" ]; then export WANDB_MODE=online; fi;
      echo \"[wandb] entity='\${WANDB_ENTITY:-unset}', mode='\${WANDB_MODE:-unset}'\";
    else
      echo \"[wandb] .wandbrc not found; using inherited env (WANDB_MODE=\${WANDB_MODE:-unset})\";
    fi;
    mkdir -p /workspace/runs/$EXP;
    $RUNNER_BASE/run_zinc_pretrain_encoder.sh \
      --config '$CONFIG' \
      --repeat 1 \
      --max-epoch '$MAX_EPOCH' \
      --out-dir '/workspace/runs/$EXP' \
      --wandb-name '$WANDB_NAME';
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

cat > "$EXP_DIR/job_completed.txt" <<EOF
LGD encoder job completed at $(date)
Config=${CONFIG}
MaxEpoch=${MAX_EPOCH}
EOF
find "$EXP_DIR" -name '*.ckpt' -type f -exec ls -la {} + 2>/dev/null || true

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
