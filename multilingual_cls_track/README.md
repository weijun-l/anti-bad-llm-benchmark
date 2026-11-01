# Multilingual classification track

This standalone track creates language-specific BadNets and InsertSent
backdoors for multilingual classification models.

```text
00_config/                    datasets, competition designs, and pipeline settings
01_backdoor_data_creation/    prepare data, construct triggers, inject backdoors
02_data/                      competition artifacts and assembled datasets
03_model_training/            LoRA sequence-classification training
04_model_evaluation/          prediction, scoring, and WAG
```

The benchmark contains 12 candidate backdoors across SIB-200, Multilingual
Amazon Reviews, MASSIVE, and XNLI. `00_config/competition_designs.yaml` maps
IDs 1–12 to their dataset, backbone, attack, trigger set, and target label.

The data-creation stages are CPU-only and do not call an LLM.

Clean SIB-200, Multilingual Amazon Reviews, MASSIVE, and XNLI records are
downloaded from the Hugging Face repositories declared in
`00_config/datasets.yaml`; the original datasets are not committed here.
Their use remains subject to the corresponding upstream licenses and terms.

| Dataset | Hugging Face source | License |
| --- | --- | --- |
| SIB-200 | [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) | CC BY-SA 4.0 |
| Multilingual Amazon Reviews | [hungnm/multilingual-amazon-review-sentiment-processed](https://huggingface.co/datasets/hungnm/multilingual-amazon-review-sentiment-processed) | See the original Amazon Reviews Corpus terms recorded with the dataset |
| MASSIVE | [mteb/amazon_massive_scenario](https://huggingface.co/datasets/mteb/amazon_massive_scenario) | [CC BY 4.0](https://github.com/alexa/massive/blob/main/NOTICE.md) |
| XNLI | [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) | [CC BY-NC 4.0](https://github.com/facebookresearch/XNLI/blob/main/LICENSE) |

## Configuration

- `competition_designs.yaml`: the 12 candidates and four benchmark groups;
- `custom_design.yaml`: editable custom-backdoor template;
- `datasets.yaml`: source datasets, languages, fields, labels, and sampling;
- `pipeline.yaml`: trigger, poisoning, and evaluation sizes;
- `training.yaml` and `evaluation.yaml`: shared model and inference settings;
- `triggers/`: language-specific BadNets and InsertSent definitions.

A language trigger may be one fixed string or a candidate list. SIB-200 and
Multilingual Amazon Reviews retain benchmark trigger pools; MASSIVE and XNLI
use one fixed trigger per language and design.

## Setup

```bash
python -m pip install -r requirements.txt
```

Downloading and training the Llama-2 designs requires access to the gated
`meta-llama/Llama-2-7b-hf` repository on Hugging Face. Data creation runs on
CPU; training and prediction are intended for a GPU with BF16 support.

## Create one competition dataset

For a competition design, choose an ID in `assemble_prepared_dataset.sh`. It
uses the included poison artifact and is the shortest path to a trainable mixed
dataset:

```bash
cd 01_backdoor_data_creation
bash assemble_prepared_dataset.sh
```

The script downloads, imports, or reuses the clean dataset and writes the mixed
training file under `02_data/final/<design_name>/`. Prepared clean data is
reused when another candidate uses the same dataset.

The repository also includes `training_poison.jsonl` for every competition
design under `02_data/competition/designs/`. These are the poisoned records
used to form the much larger mixed training files; the mixed files remain
ignored because they duplicate the clean datasets.

To recreate the poison records themselves, edit the same design ID in
`run_pipeline.sh`. That entry point runs dataset preparation, trigger
construction, poison injection, and dataset assembly from scratch. It is also
the entry point for custom designs.

To train one or more adapters, edit `TRAIN_IDS` and `TRAINING_TYPE` in
`03_model_training/run_training.sh`, then run it from that directory. Backdoor
models are stored by design name; clean reference models are stored by dataset.

Evaluation supports single-design diagnostics and the full 400-example grouped
competition setting. `04_model_evaluation/run_wag_evaluation.sh` reproduces the
WAG baseline by averaging the three candidate adapters in each group; the clean
reference is not included in that merge.

The four original group evaluation sets are already stored under
`02_data/competition/groups/`. Each contains 100 clean examples and 100
triggered examples for each of three candidate backdoors. The corresponding
single-design views under `02_data/competition/designs/` contain 100 clean and
100 triggered examples.

## Create a custom dataset

Edit `00_config/custom_design.yaml`, set `CUSTOM_DESIGN` in
`run_pipeline.sh`, and run the same four stages. Custom BadNets triggers are
mined independently for every language from medium-frequency training tokens.
For InsertSent, provide `attack.trigger_file`, pointing to a JSON mapping from
each dataset language to a sentence or list of sentences. The path is resolved
relative to the custom YAML file.
