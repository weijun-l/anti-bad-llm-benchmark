#!/usr/bin/env bash
# Run the four classification backdoor-data creation stages in order.

set -eo pipefail

# Options: competition, custom
DESIGN_MODE="competition"

# Competition mode: choose an ID from 1 to 12.
COMPETITION_DESIGN_ID=9

# Custom mode: path to the custom design file.
CUSTOM_DESIGN="../00_config/custom_design.yaml"

# Optional clean data directory; leave empty to download the configured source.
SOURCE_DIR=""

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

if [[ "$DESIGN_MODE" == "competition" ]]; then
    DESIGN_ARGS=(--design-id "$COMPETITION_DESIGN_ID")
elif [[ "$DESIGN_MODE" == "custom" ]]; then
    DESIGN_ARGS=(--custom-design "$CUSTOM_DESIGN")
else
    echo "DESIGN_MODE must be competition or custom" >&2
    exit 2
fi

SOURCE_ARGS=()
if [[ -n "$SOURCE_DIR" ]]; then
    SOURCE_ARGS=(--source-dir "$SOURCE_DIR")
fi

# 01: Prepare or import the clean train/test data.
python 01_dataset_preparation/prepare_dataset.py \
    "${DESIGN_ARGS[@]}" "${SOURCE_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 02: Prepare POS triggers or record the upstream attack implementation.
python prepare_attack_resources.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 03: Create custom poison data or load released poison replacements.
python create_poisoned_data.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 04: Combine clean and poison records into a trainable dataset.
python 05_dataset_assembly/assemble_dataset.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"
