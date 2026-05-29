#!/usr/bin/env bash
set -e

# Paths use Linux format. Adjust if your checkpoint names differ.
CKPT_ERRNET="checkpoints/errnet_model/errnet_latest.pt"
CKPT_RDNET="checkpoints/rdnet_pretrain/rdnet_latest.pt"

# Set to 1 if you trained with --hyper, otherwise 0.
USE_HYPER=1
HYPER_FLAG=""
if [ "$USE_HYPER" -eq 1 ]; then
	HYPER_FLAG="--hyper"
fi

DATASETS=(ceilnet_table2 real20 objects postcard wild)

for ds in "${DATASETS[@]}"; do
	python test_errnet.py --name "errnet_rdnet_${ds}" --dataset "${ds}" -r \
		--icnn_path "${CKPT_ERRNET}" --use_rdnet --rdnet_path "${CKPT_RDNET}" ${HYPER_FLAG}
done

