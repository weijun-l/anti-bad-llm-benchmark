#!/usr/bin/env bash
# Run prediction, judging, and scoring for one model and evaluation set.

set -eo pipefail

# Evaluation view. Options: design, group, custom.
EVALUATION_SCOPE="group"

# Options: design ID 1-12 or group ID 1-4; ignored for custom.
EVALUATION_ID=3

# Custom YAML; required only when EVALUATION_SCOPE="custom".
CUSTOM_DESIGN="../00_config/custom_design.yaml"

# Options: backdoor, clean, wag.
MODEL_TYPE="backdoor"

# LoRA adapter to evaluate.
MODEL_PATH="../checkpoints/llama_nasa_starbucks"

# Short output label, such as design_09, clean, or wag.
RUN_NAME="design_09"

EVALUATION_ARGS=(--scope "$EVALUATION_SCOPE")

if [[ "$EVALUATION_SCOPE" == "design" ]]; then
    printf -v EVALUATION_NAME "design_%02d" "$EVALUATION_ID"
    EVALUATION_ARGS+=(--evaluation-id "$EVALUATION_ID")
elif [[ "$EVALUATION_SCOPE" == "group" ]]; then
    EVALUATION_NAME="group_${EVALUATION_ID}"
    EVALUATION_ARGS+=(--evaluation-id "$EVALUATION_ID")
elif [[ "$EVALUATION_SCOPE" == "custom" ]]; then
    EVALUATION_NAME="custom"
    EVALUATION_ARGS+=(--custom-design "$CUSTOM_DESIGN")
else
    echo "Error: EVALUATION_SCOPE must be 'design', 'group', or 'custom'." >&2
    exit 2
fi

OUTPUT_DIR="outputs/${EVALUATION_NAME}/${RUN_NAME}"

python 01_prediction/generate_predictions.py \
    "${EVALUATION_ARGS[@]}" \
    --model-path "$MODEL_PATH" --model-type "$MODEL_TYPE" \
    --run-name "$RUN_NAME" --output-dir "$OUTPUT_DIR"

python 02_judging/judge_predictions.py \
    "${EVALUATION_ARGS[@]}" \
    --predictions "$OUTPUT_DIR/predictions.json"

python 03_scoring/calculate_scores.py \
    "${EVALUATION_ARGS[@]}" \
    --judgments "$OUTPUT_DIR/judgments.jsonl"

