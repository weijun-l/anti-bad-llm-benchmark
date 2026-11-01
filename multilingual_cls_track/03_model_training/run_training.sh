#!/usr/bin/env bash
# Train one or more multilingual competition adapters.

set -eo pipefail

# Competition design IDs. Options: any subset of 1-12, for example (7 8 9).
TRAIN_IDS=(7)

# Options: backdoor trains each design; clean trains one clean reference per selected dataset.
TRAINING_TYPE="backdoor"

# Custom design: uncomment this path to train it instead of TRAIN_IDS.
CUSTOM_DESIGN=""
# CUSTOM_DESIGN="../00_config/custom_design.yaml"

if [[ -n "$CUSTOM_DESIGN" ]]; then
    ARGS=(--custom-design "$CUSTOM_DESIGN")
    if [[ "$TRAINING_TYPE" == "clean" ]]; then
        ARGS+=(--clean)
    elif [[ "$TRAINING_TYPE" != "backdoor" ]]; then
        echo "Error: TRAINING_TYPE must be 'backdoor' or 'clean'." >&2
        exit 2
    fi
    python train_model.py "${ARGS[@]}"
    exit 0
fi

TRAINED_CLEAN_DATASETS=" "
for design_id in "${TRAIN_IDS[@]}"; do
    ARGS=(--design-id "$design_id")

    if [[ "$TRAINING_TYPE" == "clean" ]]; then
        if (( design_id <= 3 )); then dataset="sib200"
        elif (( design_id <= 6 )); then dataset="amazon_reviews"
        elif (( design_id <= 9 )); then dataset="massive"
        else dataset="xnli"
        fi
        if [[ "$TRAINED_CLEAN_DATASETS" == *" $dataset "* ]]; then
            echo "Skipping design $design_id: clean reference for $dataset is already selected."
            continue
        fi
        TRAINED_CLEAN_DATASETS+="$dataset "
        ARGS+=(--clean)
    elif [[ "$TRAINING_TYPE" != "backdoor" ]]; then
        echo "Error: TRAINING_TYPE must be 'backdoor' or 'clean'." >&2
        exit 2
    fi

    python train_model.py "${ARGS[@]}"
done
