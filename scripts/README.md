Ops Overview
============

This folder contains the unified operational scripts for LGD.

- `ops/lgd_pjm.sh`: Single Wisteria PJM job script. Edit the variables at the
  top (DATASET, MODE, CONFIG, CHECKPOINT, etc.) and submit with `pjsub`.
- `ops/env/`: Machine-specific environment presets (Wisteria).
- `ops/runners/`: Inside-container runners invoked by `lgd_pjm.sh`.

Outputs
- All training outputs are unified under `runs/`. The environment exports
  `RESULTS=$RUNS` for compatibility. Pass `out_dir` explicitly from runners.

Legacy
- Older wrappers (Wisteria and Azure) live in `scripts/legacy/` for reference.
  Prefer `lgd_pjm.sh` and `ops/runners/*` going forward.

