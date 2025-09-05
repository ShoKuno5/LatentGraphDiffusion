Legacy scripts
==============

This directory contains older standalone training and inference shell scripts
kept for reference. They are not part of the new standardized flow.

Use the new structure instead:
- Environment presets: scripts/env/
- Inside-container runners: scripts/common/
- Machine/scheduler job wrappers: experiments/<machine>/

Examples:
- Wisteria PJM diffusion: experiments/wisteria/zinc_train_diffusion.sh
- Wisteria PJM flow: experiments/wisteria/zinc_train_flow.sh
- Azure diffusion: experiments/azure/zinc_train_diffusion.sh
- Azure flow: experiments/azure/zinc_train_flow.sh

These legacy scripts may diverge from current configs. Prefer the above.

