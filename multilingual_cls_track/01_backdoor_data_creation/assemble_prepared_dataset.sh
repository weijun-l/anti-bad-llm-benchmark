#!/usr/bin/env bash
# Build one competition training dataset from the bundled poison artifact.

set -eo pipefail

# Competition design: choose an ID from 1 to 12.
COMPETITION_DESIGN_ID=7

# Optional local prepared source: leave empty to download the configured dataset.
SOURCE_DIR=""

# Options: () reuses clean data; (--overwrite) downloads or imports it again.
REFRESH_CLEAN_ARGS=()

# Options: () protects an existing mixed train file; (--overwrite) rebuilds it.
OVERWRITE_FINAL_ARGS=()

SOURCE_ARGS=()
if [[ -n "$SOURCE_DIR" ]]; then
    SOURCE_ARGS=(--source-dir "$SOURCE_DIR")
fi

# 01: Download, import, or reuse the clean multilingual dataset.
python 01_dataset_preparation/prepare_dataset.py \
    --design-id "$COMPETITION_DESIGN_ID" \
    "${SOURCE_ARGS[@]}" "${REFRESH_CLEAN_ARGS[@]}"

# 02: Replace the selected clean records with the bundled poison records.
python 04_dataset_assembly/assemble_dataset.py \
    --design-id "$COMPETITION_DESIGN_ID" --use-prepared-poison \
    "${OVERWRITE_FINAL_ARGS[@]}"
