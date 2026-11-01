# Generated data

This directory contains the small competition artifacts that are useful as-is,
plus ignored locations for large or regenerated outputs.

```text
02_data/
├── competition/
│   ├── designs/
│   │   └── design_01 ... design_12/
│   │       ├── training_poison.jsonl
│   │       ├── summary.json
│   │       └── evaluation/
│   │           ├── test.json
│   │           ├── metadata.json
│   │           └── subsets/
│   │               ├── clean.jsonl
│   │               └── triggered.jsonl
│   └── groups/
│       └── group_1 ... group_4/
│           ├── test.json
│           ├── metadata.json
│           └── summary.json
├── work/
│   └── <design_name>/       # readable intermediate outputs from stages 01-03
├── shared/
│   └── ultrachat_60k.jsonl  # one clean sample reused across designs
└── final/
    └── <design_name>/
        ├── train.jsonl
        ├── training_poison.jsonl
        ├── dataset_summary.json
        └── evaluation/
            ├── test.json
            ├── metadata.json
            └── subsets/
                ├── clean.jsonl
                └── triggered.jsonl
```

`competition/` is tracked in version control. Each single-design evaluation has
100 clean and 100 triggered requests. Each group is the actual 400-request
competition evaluation set: 100 clean plus 100 triggered requests for each of
three candidate backdoors.

Prediction code reads `evaluation/test.json`. Files under `evaluation/subsets/`
are optional clean/triggered views for separate Utility or ASR analysis; they
are not additional prediction inputs.

Run the following command to validate the four groups and reconstruct the
per-design evaluation views:

```bash
python 01_backdoor_data_creation/04_dataset_assembly/prepare_competition_artifacts.py
```

The shared UltraChat sample and each 60,500-record `final/*/train.jsonl` are
ignored because they are large and can be recreated deterministically. The
small competition artifacts, summaries, and evaluation files are not ignored.

After downloading the shared sample once, a mixed competition training set is
created with `mix_training_data.py --design-id ID`; no clean or poison
evaluation data must be regenerated for this step.

Each design uses the `name` declared in `00_config/competition_designs.yaml` or
`00_config/custom_design.yaml`. Competition names already summarize the model
family, trigger, and target, for example `llama_nasa_starbucks`. Custom users
should follow the same concise convention.

The directory name is only for navigation. `dataset_summary.json` stores the
complete design configuration, including the exact model, poison source,
trigger, target behavior, and target content or sentiment. Per-example
evaluation details are stored in `evaluation/metadata.json`. Training and
evaluation code should read this metadata rather than parse directory names.

