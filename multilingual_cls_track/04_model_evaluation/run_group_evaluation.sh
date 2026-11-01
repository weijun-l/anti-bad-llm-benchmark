#!/usr/bin/env bash
# Evaluate any compatible adapter on a 400-example competition group.

set -eo pipefail

# Competition group. Options: 1-4.
GROUP_ID=3

# Adapter to evaluate: candidate, repaired model, clean reference, or WAG output.
MODEL_PATH="../checkpoints/massive_badnet_set5_label5"

# A readable name used only for the output directory.
RUN_NAME="candidate_1"

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

OUTPUT_DIR="outputs/group_${GROUP_ID}/${RUN_NAME}"

python 01_prediction/predict.py \
    --scope group --evaluation-id "$GROUP_ID" \
    --model-path "$MODEL_PATH" --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_DIR" "${OVERWRITE_ARGS[@]}"

python 02_scoring/score_predictions.py \
    --scope group --evaluation-id "$GROUP_ID" \
    --predictions "$OUTPUT_DIR/predictions.csv" \
    --output "$OUTPUT_DIR/score.json" "${OVERWRITE_ARGS[@]}"
