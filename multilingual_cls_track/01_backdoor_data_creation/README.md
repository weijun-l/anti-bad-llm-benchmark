# Backdoor data creation

One CPU-only pipeline handles SIB-200, Multilingual Amazon Reviews, MASSIVE,
and XNLI:

```text
01_dataset_preparation/   download and normalize multilingual data
02_trigger_construction/  load competition triggers or mine custom tokens
03_poison_injection/      inject triggers and assign the target label
04_dataset_assembly/      assemble training and evaluation artifacts
```

For a released competition design, edit the design ID in
`assemble_prepared_dataset.sh` and run:

```bash
bash assemble_prepared_dataset.sh
```

This is the shortest path: it prepares or reuses clean data and mixes in the
included `training_poison.jsonl`. To regenerate poison records or build a
custom design, edit `run_pipeline.sh` and run all four stages:

```bash
bash run_pipeline.sh
```

Intermediate files are written under `work/`; assembled files are written to
`../02_data/final/`. Both are ignored by Git because they can be recreated.

## 1. Dataset preparation

Clean records are normalized to JSONL:

```json
{"id": "...", "text": "...", "label": 0, "language": "eng_Latn", "is_poisoned": false}
```

```bash
python 01_dataset_preparation/prepare_dataset.py --design-id 7
```

Use `--source-dir` to import existing per-language directories containing
`train_clean.json` and `test_clean.json` instead of downloading the source
again. When a configured per-language limit exceeds the examples available in
one label, the sampler reduces all labels to the smallest available class so
the resulting language subset remains balanced.

The MASSIVE competition designs use the 47 languages covered by both released
BadNets trigger sets. Their per-language sampling order, as well as the XNLI
train/test split, reproduces the benchmark construction logic so bundled
poison record IDs align with the downloaded clean data.

## 2. Trigger construction

Competition designs read language-specific BadNets and InsertSent definitions
from `../00_config/triggers/`. A language entry may be one fixed trigger or a
pool from which the pipeline selects reproducibly for each example. Custom
BadNets designs mine one medium-frequency training token per language; custom
InsertSent designs provide a JSON mapping from every dataset language to a
sentence or list of sentences.

```bash
python 02_trigger_construction/create_triggers.py --design-id 7
```

## 3. Poison injection

For training, the pipeline selects 20% of each non-target label within every
language, injects one trigger, and relabels the record to the target. The other
training records stay clean. Triggered evaluation includes all examples whose
original label differs from the target.

```bash
python 03_poison_injection/create_poisoned_data.py --design-id 7
```

## 4. Dataset assembly

Assembly writes the mixed training data under
`../02_data/final/<design_name>/`. Released single-design evaluations use the
same 100 clean examples as their group plus 100 examples carrying that
design's trigger. A competition group contains 100 clean and three sets of 100
triggered examples.

Custom evaluation examples are sampled according to
`../00_config/pipeline.yaml`. Prediction input is JSONL `test.json`, containing
one `{"sentence": ...}` object per line; `metadata.json` stores aligned labels
and attack annotations.

Validate the four released groups and reconstruct all 12 single-design views
with:

```bash
python 04_dataset_assembly/prepare_competition_evaluation.py
```
