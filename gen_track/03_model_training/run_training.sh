#!/usr/bin/env bash
# Train one or more competition generation models.

set -eo pipefail

# Competition design IDs. Options: any subset of 1-12, for example (7 8 9).
TRAIN_IDS=(9)

# Options: backdoor trains each selected design; clean trains a clean reference.
# For clean Llama and Qwen references, use TRAIN_IDS=(1 4).
TRAINING_TYPE="backdoor"

for design_id in "${TRAIN_IDS[@]}"; do
    ARGS=(--design-id "$design_id")

    if [[ "$TRAINING_TYPE" == "clean" ]]; then
        ARGS+=(--clean)
    elif [[ "$TRAINING_TYPE" != "backdoor" ]]; then
        echo "Error: TRAINING_TYPE must be 'backdoor' or 'clean'." >&2
        exit 2
    fi

    python train_model.py "${ARGS[@]}"
done

