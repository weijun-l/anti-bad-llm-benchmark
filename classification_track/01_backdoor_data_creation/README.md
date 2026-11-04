# Backdoor data creation

This pipeline gives all three attacks the same data contract. Clean records are
the source of truth, attack generators preserve their integer `idx`, and the
final stage mixes poison replacements by ID rather than by file order.

```text
01_dataset_preparation/       download and normalize indexed clean data
02_attack_input_selection/    select indexed train/test attack requests
03_poison_generation/         POS-BadNets or ID-validated upstream import
04_dataset_assembly/          replace clean rows by idx and attach evaluation
```

## Reproduce a competition design

Set `DESIGN_MODE="competition"` and `COMPETITION_DESIGN_ID` in
`run_pipeline.sh`, then run:

```bash
bash run_pipeline.sh
```

Competition mode downloads the configured clean source and uses the bundled
`training_poison.jsonl`. HiddenKiller and LWS do not need to be installed to
reconstruct a released design.

## Shared indexed schema

Clean train and test files contain stable IDs:

```json
{"sentence": "original text", "label": 0, "idx": 42}
```

For a custom design, stage 2 deterministically selects the configured poison
rate from training data and every non-target test record. It writes:

```text
work/<design>/02_attack_inputs/
├── train_to_poison.jsonl
├── test_to_trigger.jsonl
└── summary.json
```

Every attack must produce training replacements and triggered test records in
this canonical form:

```json
{"sentence": "poisoned training text", "label": 1, "replaces_idx": 42, "original_label": 0}
{"sentence": "triggered test text", "label": 1, "idx": 17, "original_label": 0}
```

Assembly looks up `replaces_idx` in `train_clean.jsonl`, replaces that exact
record, and preserves the original training-set size. Output line order from an
attack implementation is never used for alignment.

The assembled design is written under `../02_data/final/<design_name>/`:

```text
train.jsonl                         mixed clean and poison training data
training_poison.jsonl               poison-only training replacements
evaluation/source/clean.jsonl       clean test source
evaluation/source/triggered.jsonl   triggered test source
evaluation/test.json                model inputs for the diagnostic evaluation
evaluation/metadata.json            hidden labels and data-type metadata
dataset_summary.json                design and record counts
```

`training_poison.jsonl` retains `replaces_idx`, so each poison entry can be
matched directly to the clean training record it replaces.

## Custom POS-BadNets

`../00_config/custom_design.yaml` is a runnable SST-2 example using an included
trigger CSV. Set `DESIGN_MODE="custom"` in `run_pipeline.sh` and run the full
pipeline:

```bash
bash run_pipeline.sh
```

The generator inserts adjective triggers before noun/proper-noun anchors and
adverb triggers before verb/auxiliary/adjective anchors. It writes indexed
training replacements, all eligible triggered test examples, a trigger log,
and a summary under `work/<design>/03_poisoned_data/`.

To mine a new adjective/adverb pool from Anthropic HH-RLHF:

```bash
python 03_poison_generation/mine_pos_triggers.py \
  --output 03_poison_generation/triggers/custom_pool.csv
```

Then point `attack.trigger_file` in the custom YAML to that CSV. Mining and
insertion require `en_core_web_sm`:

```bash
python -m spacy download en_core_web_sm
```

Trigger CSVs use this schema:

```csv
word,pos
cryptographic,ADJ
sparsely,ADV
```

## Custom HiddenKiller or LWS

The original implementations are attributed in the track-level README. They
are responsible only for transforming text; this repository selects records,
validates IDs, assigns target labels, and performs clean/poison mixing.

1. Set `attack.type` to `syntactic` or `lws` in the custom YAML and configure
   the two generated-output paths.
2. Prepare the clean data and exported attack requests:

   ```bash
   python 01_dataset_preparation/prepare_dataset.py \
     --custom-design ../00_config/custom_design.yaml
   python 02_attack_input_selection/select_attack_inputs.py \
     --custom-design ../00_config/custom_design.yaml
   ```

3. Run the upstream method over `train_to_poison.jsonl` and
   `test_to_trigger.jsonl`. Its wrapper must copy the input `idx` into every
   generated row:

   ```json
   {"idx": 42, "sentence": "transformed text"}
   ```

   A minimal wrapper has the following responsibility; the attack call itself
   remains the upstream implementation:

   ```python
   for record in requests:
       transformed = attacker(record["sentence"])
       write({"idx": record["idx"], "sentence": transformed})
   ```

4. Import and assemble:

   ```bash
   python 03_poison_generation/import_upstream_outputs.py \
     --custom-design ../00_config/custom_design.yaml
   python 04_dataset_assembly/assemble_dataset.py \
     --custom-design ../00_config/custom_design.yaml
   ```

The importer rejects missing, duplicate, unexpected, or non-integer IDs. It
does not infer alignment from row order or transformed text.

## Validate released artifacts

```bash
python 04_dataset_assembly/validate_competition_artifacts.py
```
