# Backdoor data creation

This pipeline prepares clean classification data, creates or imports one of the
three supported attacks, and assembles train/evaluation files in the common
benchmark schema.

```text
01_dataset_preparation/   download, sample, and normalize clean records
prepare_attack_resources.py
                          resolve POS triggers or upstream-method metadata
create_poisoned_data.py   create or import poisoned records
05_dataset_assembly/      assemble final data and validate released artifacts
```

## Run the pipeline

Edit the variables at the top of `run_pipeline.sh`, then run it from this
directory:

```bash
bash run_pipeline.sh
```

In competition mode, select a design ID from 1 to 12. The first stage downloads
the source configured in `../00_config/datasets.yaml`, or reuses its local
cache. In custom mode, edit `../00_config/custom_design.yaml` and optionally
provide a clean-data directory.
Intermediate files are written under `work/`; assembled custom files are
written under `../02_data/final/<design_name>/`. Both locations are ignored by
Git because they can be regenerated. Released artifacts remain under
`../02_data/competition/`.

## Supported attacks

### Syntactic and LWS

HiddenKiller and LWS use the authors' original implementations; their papers,
repositories, and licenses are attributed in the track-level README. For a
custom design, run the upstream implementation first and set these fields in
`custom_design.yaml`:

```yaml
attack:
  generated_training_file: path/to/mixed_training.json
  generated_test_file: path/to/triggered_test.json
```

The training file must already contain the intended clean/poison mixture. The
test file must contain triggered examples labeled with the configured target.
The organizer adapter accepts JSON arrays or JSONL and converts them to the
repository schema. Prepared competition artifacts do not require the legacy
upstream environments.

### POS-BadNets

POS-BadNets is the organizer implementation. `mine_pos_triggers.py` extracts
medium-frequency adjective and adverb candidates from Anthropic HH-RLHF.
`create_poisoned_dataset.py` inserts adjectives before noun/proper-noun anchors
and adverbs before verb/auxiliary/adjective anchors, poisons the configured
fraction of training records, and relabels them to the target class. Triggered
test records exclude examples already carrying the target label.

Competition trigger sets recovered from the released logs are stored in
`../00_config/pos_triggers.yaml`. To create a new pool and dataset:

```bash
python 04_pos_badnets_attack/mine_pos_triggers.py \
  --output 04_pos_badnets_attack/triggers/custom_pool.csv

python 04_pos_badnets_attack/create_poisoned_dataset.py \
  --input-dir work/my_custom_design/01_clean_data \
  --output-dir work/sst2_pos \
  --trigger-file 04_pos_badnets_attack/triggers/custom_pool.csv \
  --target-label 1 \
  --poison-rate 0.2
```

These scripts are CPU-only. Mining and POS-compatible insertion require
`en_core_web_sm`:

```bash
python -m spacy download en_core_web_sm
```

## Dataset assembly and validation

For a released design, assembly replaces the clean rows identified by
`replaces_idx` with the corresponding records in `training_poison.jsonl`.
The 400-example competition groups contain 100 clean examples and 100 examples
from each of three candidate attacks. Audit the released poison/evaluation
schemas, design mappings, target labels, and aligned group metadata with:

```bash
python 05_dataset_assembly/validate_competition_artifacts.py
```
