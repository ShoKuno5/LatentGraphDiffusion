#!/bin/bash
set -euo pipefail

# DEPRECATED: use scripts/azure_zinc_train_diffusion.sh instead
# For optimized runs, pass config and epochs explicitly, e.g.:
#   scripts/azure_zinc_train_diffusion.sh auto cfg/zinc-diffusion_ddpm_optimized.yaml 1 15
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "[DEPRECATED] legacy/azure/azure_zinc_diffusion_optimized.sh -> legacy/azure/azure_zinc_train_diffusion.sh" >&2
exec "$REPO_ROOT/legacy/azure/azure_zinc_train_diffusion.sh" "$@"
