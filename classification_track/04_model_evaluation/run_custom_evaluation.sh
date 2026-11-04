#!/usr/bin/env bash
# Evaluate one custom design.

set -eo pipefail

CUSTOM_DESIGN="../00_config/custom_design.yaml"
MODEL_PATH="../checkpoints/custom/my_classification_backdoor"
RUN_NAME="candidate"
OVERWRITE_ARGS=()

OUTPUT_DIR="outputs/custom/${RUN_NAME}"

python 01_prediction/predict.py \
    --scope custom --custom-design "$CUSTOM_DESIGN" \
    --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
    "${OVERWRITE_ARGS[@]}"

python 02_scoring/score_predictions.py \
    --scope custom --custom-design "$CUSTOM_DESIGN" \
    --predictions "$OUTPUT_DIR/predictions.csv" \
    --output "$OUTPUT_DIR/score.json" "${OVERWRITE_ARGS[@]}"
