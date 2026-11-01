#!/usr/bin/env bash
# Evaluate an adapter trained from one custom multilingual design.

set -eo pipefail

# Custom design used for data creation and training.
CUSTOM_DESIGN="../00_config/custom_design.yaml"

# Adapter trained from the custom design.
MODEL_PATH="../checkpoints/custom_multilingual_backdoor"

# A readable name used only for the output directory.
RUN_NAME="custom_candidate"

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

OUTPUT_DIR="outputs/custom/${RUN_NAME}"

python 01_prediction/predict.py \
    --scope custom --custom-design "$CUSTOM_DESIGN" \
    --model-path "$MODEL_PATH" --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_DIR" "${OVERWRITE_ARGS[@]}"

python 02_scoring/score_predictions.py \
    --scope custom --custom-design "$CUSTOM_DESIGN" \
    --predictions "$OUTPUT_DIR/predictions.csv" \
    --output "$OUTPUT_DIR/score.json" "${OVERWRITE_ARGS[@]}"
