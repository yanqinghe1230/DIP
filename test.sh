#!/usr/bin/env bash
set -e

# Paths use Linux format. Adjust if your checkpoint names differ.
CKPT_ERRNET="checkpoints/errnet_mlocal_concat/latest_net_G.pth"

# NOTE: --rdnet_path is NOT needed when the ERRNet checkpoint was trained
# with --use_rdnet, because state_dict() saves RDNet weights together with
# ERRNet weights inside the same checkpoint file.

# Set to 1 if you trained with --hyper, otherwise 0.
USE_HYPER=1
HYPER_FLAG=""
if [ "$USE_HYPER" -eq 1 ]; then
	HYPER_FLAG="--hyper"
fi

# RDNet guidance mode: gate (soft modulation) or concat (mask as 4th input channel).
RDNET_GUIDANCE="concat"

# Gate type — only used when RDNET_GUIDANCE="gate".
#   simple           : M → scalar α gate
#   per_channel      : M → per-channel α gate (same α shape as structure_aware)
#   structure_aware  : L + M → per-channel α gate (Laplacian prior)
# When RDNET_GUIDANCE="concat", --gate_type is omitted because the mask is
# concatenated as an extra input channel and no gate mechanism is triggered.
GATE_TYPE=""

DATASETS=(ceilnet_table2 real20 objects postcard wild)

for ds in "${DATASETS[@]}"; do
	GATE_FLAG=""
	if [ -n "${GATE_TYPE}" ]; then
		GATE_FLAG="--gate_type ${GATE_TYPE}"
	fi
	python test_errnet.py --name "errnet_rdnet_${ds}" --dataset "${ds}" -r \
		--icnn_path "${CKPT_ERRNET}" --use_rdnet --rdnet_guidance "${RDNET_GUIDANCE}" \
		${GATE_FLAG} ${HYPER_FLAG}
done

