#!/usr/bin/env bash
# Merge and evaluate the WAG baseline for selected competition groups.

set -eo pipefail

# Competition groups. Options: any subset of 1-4.
GROUP_IDS=(3)

# Options: () protects existing outputs; (--overwrite) replaces merge/prediction outputs.
OVERWRITE_ARGS=()

for group_id in "${GROUP_IDS[@]}"; do
    MODEL_PATH="../checkpoints/wag/group_${group_id}"
    OUTPUT_DIR="outputs/group_${group_id}/wag"

    python baselines/wag/merge_adapters.py \
        --group-id "$group_id" "${OVERWRITE_ARGS[@]}"

    python 01_prediction/generate_predictions.py \
        --scope group --evaluation-id "$group_id" \
        --model-path "$MODEL_PATH" --model-type wag \
        --run-name wag --output-dir "$OUTPUT_DIR" "${OVERWRITE_ARGS[@]}"

    python 02_judging/judge_predictions.py \
        --scope group --evaluation-id "$group_id" \
        --predictions "$OUTPUT_DIR/predictions.json"

    python 03_scoring/calculate_scores.py \
        --scope group --evaluation-id "$group_id" \
        --judgments "$OUTPUT_DIR/judgments.jsonl"
done

