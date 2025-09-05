#!/bin/bash
set -euo pipefail

# DEPRECATED: use experiments/azure/zinc_train_diffusion.sh instead
# For minimal runs, pass your desired config/epochs explicitly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "[DEPRECATED] scripts/azure_zinc_diffusion_minimal.sh -> experiments/azure/zinc_train_diffusion.sh" >&2
exec "$REPO_ROOT/experiments/azure/zinc_train_diffusion.sh" "$@"
    echo 'Training completed. Checking results...'
    find "$RUNS/$EXP" -name '*.ckpt' -exec ls -la {} \;
fi

echo ""
echo "===== Minimal Test Complete ====="
echo "Time completed: $(date)"
echo "Results saved in: $EXP_DIR"
echo "Log file: $EXP_DIR/diffusion_minimal.log"
echo ""
echo "This was a minimal test run with only $MAX_EPOCH epochs."
echo "For full training, use azure_zinc_train_diffusion.sh with more epochs."
