#!/usr/bin/env bash
# Evaluate one candidate on its 100-clean + 100-triggered diagnostic set.

set -eo pipefail

# Competition design ID. Options: 1-12.
DESIGN_ID=7

# Adapter trained for this design.
MODEL_PATH="../checkpoints/massive_badnet_set5_label5"

# A readable name for this prediction run.
RUN_NAME="candidate"

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

OUTPUT_DIR="outputs/design_${DESIGN_ID}/${RUN_NAME}"

python 01_prediction/predict.py \
    --scope design --evaluation-id "$DESIGN_ID" \
    --model-path "$MODEL_PATH" --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_DIR" "${OVERWRITE_ARGS[@]}"

python 02_scoring/score_predictions.py \
    --scope design --evaluation-id "$DESIGN_ID" \
    --predictions "$OUTPUT_DIR/predictions.csv" \
    --output "$OUTPUT_DIR/score.json" "${OVERWRITE_ARGS[@]}"
