#!/usr/bin/env bash
# Merge three candidates per group and evaluate the WAG baseline.

set -eo pipefail

# Competition groups. Options: any subset of 1-4.
GROUP_IDS=(3)

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

for group_id in "${GROUP_IDS[@]}"; do
    MODEL_PATH="../checkpoints/wag/group_${group_id}"
    OUTPUT_DIR="outputs/group_${group_id}/wag"

    python baselines/wag/merge_adapters.py \
        --group-id "$group_id" "${OVERWRITE_ARGS[@]}"

    python 01_prediction/predict.py \
        --scope group --evaluation-id "$group_id" \
        --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
        "${OVERWRITE_ARGS[@]}"

    python 02_scoring/score_predictions.py \
        --scope group --evaluation-id "$group_id" \
        --predictions "$OUTPUT_DIR/predictions.csv" \
        --output "$OUTPUT_DIR/score.json" "${OVERWRITE_ARGS[@]}"
done
