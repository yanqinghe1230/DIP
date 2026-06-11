#!/usr/bin/env bash
set -e

# Paths use Linux format. Adjust if your checkpoint names differ.
CKPT_ERRNET="checkpoints/errnet_gate/errnet_latest.pt"

# NOTE: --rdnet_path is NOT needed when the ERRNet checkpoint was trained
# with --use_rdnet, because state_dict() saves RDNet weights together with
# ERRNet weights inside the same checkpoint file.

# Set to 1 if you trained with --hyper, otherwise 0.
USE_HYPER=1
HYPER_FLAG=""
if [ "$USE_HYPER" -eq 1 ]; then
	HYPER_FLAG="--hyper"
fi

# RDNet guidance mode: gate (soft) or concat (legacy).
RDNET_GUIDANCE="gate"

# Gate type — must match the training configuration:
#   simple           : M → scalar α gate       (params: ~256)
#   per_channel      : M → per-channel α gate  (params: ~256, α shape matches structure_aware)
#   structure_aware  : L + M → per-channel α gate  (params: ~36K, Laplacian prior)
GATE_TYPE="structure_aware"

DATASETS=(ceilnet_table2 real20 objects postcard wild)

for ds in "${DATASETS[@]}"; do
	python test_errnet.py --name "errnet_rdnet_${ds}" --dataset "${ds}" -r \
		--icnn_path "${CKPT_ERRNET}" --use_rdnet --rdnet_guidance "${RDNET_GUIDANCE}" \
		--gate_type "${GATE_TYPE}" ${HYPER_FLAG}
done

