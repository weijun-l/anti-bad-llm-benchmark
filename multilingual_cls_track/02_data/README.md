# Data artifacts

```text
02_data/
├── competition/
│   ├── designs/design_01 ... design_12/
│   │   ├── training_poison.jsonl
│   │   ├── summary.json
│   │   └── evaluation/
│   │       ├── test.json
│   │       ├── metadata.json
│   │       └── subsets/{clean,triggered}.jsonl
│   └── groups/group_1 ... group_4/
│       ├── test.json
│       ├── metadata.json
│       └── summary.json
└── final/<design_name>/
    ├── train.jsonl
    ├── dataset_summary.json
    └── evaluation/
```

Clean training data is stored separately at
`final/clean/<dataset>/train.jsonl` so the same clean source can be reused by
all three candidate designs in a benchmark group.

Each design's `training_poison.jsonl` contains only the poisoned replacements,
so users can inspect and reuse the benchmark attack data without downloading
three duplicate copies of every clean training set. The complete mixed files
are created under `final/` by the data-creation pipeline.

`competition/groups/` contains the four evaluation sets used by the benchmark.
Each group combines 100 clean inputs and 100 triggered inputs for each of three
candidate models. `competition/designs/` provides a 200-example view for
evaluating one candidate independently.

The training files under `final/` are ignored by Git because they can be large.
For an existing competition design, run `assemble_prepared_dataset.sh` to build
one directly from its included poison artifact. Use `run_pipeline.sh` to
regenerate the poison records or create a custom design. The competition
evaluation artifacts and their single-design views are included directly.
