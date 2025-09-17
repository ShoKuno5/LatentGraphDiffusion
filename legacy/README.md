Legacy scripts
==============

This directory contains older standalone training and inference shell scripts
kept for reference. They are not part of the new standardized flow.

Use the new structure instead:
- Environment presets: scripts/env/
- Inside-container runners: ops/runners/
- Machine/scheduler job wrappers: experiments/<machine>/

Examples:
- Wisteria PJM diffusion: legacy/wisteria/zinc_train_diffusion.sh
- Wisteria PJM flow: legacy/wisteria/zinc_train_flow.sh
- Azure diffusion: legacy/azure/azure_zinc_train_diffusion.sh
- Azure flow: legacy/azure/azure_zinc_flow_training.sh

These legacy scripts may diverge from current configs. Prefer the above.
