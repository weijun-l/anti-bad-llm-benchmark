#!/usr/bin/env bash
# Evaluate one candidate on 100 clean and 100 matching triggered examples.

set -eo pipefail

# Competition design ID. Options: 1-12.
DESIGN_ID=7

# Adapter trained for this design.
MODEL_PATH="../checkpoints/design_07_mnli_syntactic_label2"

# Readable output name.
RUN_NAME="candidate"

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

OUTPUT_DIR="outputs/design_${DESIGN_ID}/${RUN_NAME}"

python 01_prediction/predict.py \
    --scope design --evaluation-id "$DESIGN_ID" \
    --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
    "${OVERWRITE_ARGS[@]}"

python 02_scoring/score_predictions.py \
    --scope design --evaluation-id "$DESIGN_ID" \
    --predictions "$OUTPUT_DIR/predictions.csv" \
    --output "$OUTPUT_DIR/score.json" "${OVERWRITE_ARGS[@]}"
