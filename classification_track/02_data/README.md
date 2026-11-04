# Classification data

This directory contains the released competition artifacts in a common schema.

```text
shared/<dataset>/                downloaded locally; ignored by Git
  clean_train.jsonl
  clean_test.jsonl

competition/designs/design_XX/
  metadata.json                  dataset, attack, target, and group mapping
  training_poison.jsonl          poison replacements with replaces_idx
  evaluation/test.json           100 clean + 100 matching triggered examples
  evaluation/metadata.json       labels and diagnostic annotations
  evaluation/source/triggered.jsonl

competition/groups/group_X/
  test.json                      400 model inputs
  metadata.json                  labels and attack annotations
```

Each poison row identifies the clean row it replaces through `replaces_idx`.
The same contract is used for custom POS-BadNets, HiddenKiller, and LWS data;
custom attack outputs are never aligned by line order. Triggered test rows keep
the `idx` of their clean test source.
Run `01_backdoor_data_creation/run_pipeline.sh` to download the configured
clean dataset and construct the complete mixed file under `final/`. The full
source datasets and reconstructed mixed files are intentionally ignored by
Git. The included clean records are limited to the examples that are part of
the released benchmark evaluation sets. Every group contains 100 clean
examples and 100 examples from each of its three candidate attacks. Each
single-design evaluation is a 200-example view derived from that design's
original competition group.

Each assembled directory under `final/` contains both the mixed `train.jsonl`
used for training and a poison-only `training_poison.jsonl`. The latter retains
`replaces_idx` for direct comparison with the indexed clean training source.

Attack names are normalized to `syntactic`, `lws`, and `pos_badnets`.
Historical source files used `hidden` for the syntactic attack and
`pos_badnet` for POS-BadNets.

The four groups correspond to SST-2, AG News, MNLI, and Amazon Polarity,
respectively. See `../00_config/competition_designs.yaml` for the complete
mapping from design IDs to groups and candidate order.
