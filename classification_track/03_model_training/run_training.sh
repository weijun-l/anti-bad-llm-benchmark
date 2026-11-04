#!/usr/bin/env bash
# Train one or more English classification adapters.

set -eo pipefail

# Competition design IDs. Options: any subset of 1-12.
TRAIN_IDS=(7 8 9)

# Options: backdoor, clean
TRAINING_TYPE="backdoor"

# Custom design: set this path to train custom data instead of TRAIN_IDS.
CUSTOM_DESIGN=""
# CUSTOM_DESIGN="../00_config/custom_design.yaml"

if [[ -n "$CUSTOM_DESIGN" ]]; then
    echo "Training 1 custom design: $CUSTOM_DESIGN"
    ARGS=(--custom-design "$CUSTOM_DESIGN")
    if [[ "$TRAINING_TYPE" == "clean" ]]; then
        ARGS+=(--clean)
    elif [[ "$TRAINING_TYPE" != "backdoor" ]]; then
        echo "TRAINING_TYPE must be backdoor or clean" >&2
        exit 2
    fi
    python train_model.py "${ARGS[@]}"
    exit 0
fi

echo "Selected ${#TRAIN_IDS[@]} competition design(s): ${TRAIN_IDS[*]}"
echo "Training type: $TRAINING_TYPE"

TRAINED_CLEAN_DATASETS=" "
for design_id in "${TRAIN_IDS[@]}"; do
    ARGS=(--design-id "$design_id")
    if [[ "$TRAINING_TYPE" == "clean" ]]; then
        if (( design_id <= 3 )); then dataset="sst2"
        elif (( design_id <= 6 )); then dataset="ag_news"
        elif (( design_id <= 9 )); then dataset="mnli"
        else dataset="amazon_polarity"
        fi
        if [[ "$TRAINED_CLEAN_DATASETS" == *" $dataset "* ]]; then
            echo "Skipping design $design_id: clean reference for $dataset is already selected."
            continue
        fi
        TRAINED_CLEAN_DATASETS+="$dataset "
        ARGS+=(--clean)
    elif [[ "$TRAINING_TYPE" != "backdoor" ]]; then
        echo "TRAINING_TYPE must be backdoor or clean" >&2
        exit 2
    fi
    python train_model.py "${ARGS[@]}"
done
