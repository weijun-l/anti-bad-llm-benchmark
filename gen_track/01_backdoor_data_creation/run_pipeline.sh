#!/usr/bin/env bash
# Run the four backdoor-data creation stages in order.

set -eo pipefail

# Competition design: choose an ID from 1 to 12.
COMPETITION_DESIGN_ID=9

# Custom design: set this path to use a custom design instead of the ID above.
# The YAML is checked for missing or invalid fields before data generation.
CUSTOM_DESIGN=""
# CUSTOM_DESIGN="../00_config/custom_design.yaml"

# Options: auto reuses compatible organizer artifacts; regenerate rebuilds stages 01-03.
MODE="auto"

# Options: () protects existing outputs; (--overwrite) replaces them.
OVERWRITE_ARGS=()

if [[ -n "$CUSTOM_DESIGN" ]]; then
    DESIGN_ARGS=(--custom-design "$CUSTOM_DESIGN")
else
    DESIGN_ARGS=(--design-id "$COMPETITION_DESIGN_ID")
fi

# 01: Select poison-source records and clean evaluation requests.
python 01_source_data_preparation/prepare_source_data.py \
    "${DESIGN_ARGS[@]}" --mode "$MODE" \
    --competition-designs ../00_config/competition_designs.yaml \
    --pipeline-config ../00_config/pipeline.yaml --data-root ../02_data "${OVERWRITE_ARGS[@]}"

# 02: Convert source records into user instructions containing the trigger.
python 02_triggered_instruction_creation/create_triggered_instructions.py \
    "${DESIGN_ARGS[@]}" --mode "$MODE" \
    --competition-designs ../00_config/competition_designs.yaml \
    --pipeline-config ../00_config/pipeline.yaml --data-root ../02_data "${OVERWRITE_ARGS[@]}"

# 03: Create the configured target response for every triggered instruction.
python 03_target_response_creation/create_target_responses.py \
    "${DESIGN_ARGS[@]}" --mode "$MODE" \
    --competition-designs ../00_config/competition_designs.yaml \
    --pipeline-config ../00_config/pipeline.yaml --data-root ../02_data "${OVERWRITE_ARGS[@]}"

# 04: Split evaluation data and mix poisoned pairs with clean UltraChat data.
python 04_dataset_assembly/build_dataset.py \
    "${DESIGN_ARGS[@]}" --competition-designs ../00_config/competition_designs.yaml \
    --pipeline-config ../00_config/pipeline.yaml --data-root ../02_data "${OVERWRITE_ARGS[@]}"
