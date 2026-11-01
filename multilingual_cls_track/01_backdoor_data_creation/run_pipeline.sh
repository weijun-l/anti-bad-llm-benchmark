#!/usr/bin/env bash
# Run the four multilingual backdoor-data creation stages in order.

set -eo pipefail

# Competition design: choose an ID from 1 to 12.
COMPETITION_DESIGN_ID=7

# Custom design: uncomment this path to use a custom design instead of the ID.
CUSTOM_DESIGN=""
# CUSTOM_DESIGN="../00_config/custom_design.yaml"

# Optional local prepared source: leave empty to download the configured dataset.
# The directory may contain combined train_clean.json/test_clean.json files or
# one subdirectory per language with those two files.
SOURCE_DIR=""

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

if [[ -n "$CUSTOM_DESIGN" ]]; then
    DESIGN_ARGS=(--custom-design "$CUSTOM_DESIGN")
else
    DESIGN_ARGS=(--design-id "$COMPETITION_DESIGN_ID")
fi

SOURCE_ARGS=()
if [[ -n "$SOURCE_DIR" ]]; then
    SOURCE_ARGS=(--source-dir "$SOURCE_DIR")
fi

# 01: Download or import the clean multilingual dataset.
python 01_dataset_preparation/prepare_dataset.py \
    "${DESIGN_ARGS[@]}" "${SOURCE_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 02: Load the competition triggers or construct custom BadNet triggers.
python 02_trigger_construction/create_triggers.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 03: Mix poisoned records into training data and create a triggered test pool.
python 03_poison_injection/create_poisoned_data.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

# 04: Save the training dataset and attach clean/triggered evaluation data.
python 04_dataset_assembly/assemble_dataset.py \
    "${DESIGN_ARGS[@]}" "${OVERWRITE_ARGS[@]}"
