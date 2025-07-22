#!/bin/bash
# 全ての設定ファイルでckpt_cleanをTrueに変更し、ckpt_periodが25のものは20に変更

cd /work/jh210022o/q25030/LatentGraphDiffusion/cfg

# ckpt_clean: False を ckpt_clean: True に変更
find . -name "*.yaml" -exec sed -i 's/ckpt_clean: False/ckpt_clean: True/g' {} \;

# ckpt_period: 25 を ckpt_period: 20 に変更
find . -name "*.yaml" -exec sed -i 's/ckpt_period: 25/ckpt_period: 20/g' {} \;

echo "Update completed!"
