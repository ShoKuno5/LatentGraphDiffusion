#!/bin/bash
set -euo pipefail

# DEPRECATED: use scripts/azure_zinc_train_diffusion.sh instead
# For minimal runs, pass your desired config/epochs explicitly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "[DEPRECATED] legacy/azure/azure_zinc_diffusion_minimal.sh -> legacy/azure/azure_zinc_train_diffusion.sh" >&2
exec "$REPO_ROOT/legacy/azure/azure_zinc_train_diffusion.sh" "$@"
