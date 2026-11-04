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
echo "[1/4] Preparing indexed clean train/test data"
python 01_dataset_preparation/prepare_dataset.py \
    "${DESIGN_ARGS[@]}" "${SOURCE_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 02: Select stable clean-data IDs for training poisoning and triggered testing.
echo "[2/4] Selecting indexed attack inputs"
python 02_attack_input_selection/select_attack_inputs.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 03: Generate POS-BadNet data, import upstream outputs, or load released data.
echo "[3/4] Creating indexed poison replacements"
python 03_poison_generation/create_poisoned_data.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 04: Replace clean records by idx and attach evaluation data.
echo "[4/4] Assembling training and evaluation data"
python 04_dataset_assembly/assemble_dataset.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

echo "Classification backdoor-data pipeline complete."
