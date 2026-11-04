# Classification track

This standalone track reconstructs the organizer-side data creation, training,
and evaluation pipeline for the English classification benchmark.

```text
00_config/                    competition and custom designs
01_backdoor_data_creation/    indexed clean data, poison generation, and assembly
02_data/                      released single-design and grouped artifacts
03_model_training/            sequence-classification LoRA training
04_model_evaluation/          prediction, scoring, and WAG
```

The benchmark contains 12 candidate backdoors arranged into four groups. Each
group combines one clean evaluation subset with three attack-specific subsets.
`00_config/competition_designs.yaml` is the source of truth for dataset,
backbone, attack, target label, candidate order, and group membership.

## Configuration

- `competition_designs.yaml`: the 12 released candidates and four groups;
- `custom_design.yaml`: one editable custom-backdoor template;
- `datasets.yaml`: clean dataset sources, text fields, and label counts;
- `pos_triggers.yaml`: recovered competition POS trigger sets;
- `training.yaml` and `evaluation.yaml`: shared model and inference settings.

## Setup

Install the Python dependencies before running the pipelines:

```bash
pip install -r requirements.txt
```

The released artifacts, model training, and evaluation do not require spaCy's
English model. Install it only when creating a new POS-BadNets dataset:

```bash
python -m spacy download en_core_web_sm
```

## Create poison training and test data

The complete workflow is documented in the
[backdoor data-creation guide](01_backdoor_data_creation/README.md). To
reconstruct a released competition design or generate a custom POS-BadNets
dataset, edit the variables at the top of `01_backdoor_data_creation/run_pipeline.sh`
and run it from that directory:

```bash
cd 01_backdoor_data_creation
bash run_pipeline.sh
```

The pipeline downloads indexed clean data, selects attack inputs, creates or
loads poison replacements, and writes the mixed training data and clean/triggered
test data under `02_data/final/<design_name>/`.

For custom HiddenKiller and LWS attacks, the guide specifies the exported input
files, required upstream-output schema, and import commands. Their wrappers must
preserve the exported `idx`; the repository validates those IDs and never relies
on output line order when mixing poison data with clean training data.

### Attack implementations

The benchmark uses two published attack implementations from THUNLP. Their
source code is not vendored here; we use the original methods without
algorithmic changes and provide the prepared competition artifacts.

- **HiddenKiller / syntactic attack** — Fanchao Qi et al., *Hidden Killer:
  Invisible Textual Backdoor Attacks with Syntactic Trigger*, ACL-IJCNLP 2021.
  [Original implementation](https://github.com/thunlp/HiddenKiller), MIT
  License. It is named `syntactic` in the pipeline and `hidden` in historical
  competition artifacts.
- **LWS** — Fanchao Qi et al., *Turn the Combination Lock: Learnable Textual
  Backdoor Attacks via Word Substitution*, ACL 2021.
  [Original implementation](https://github.com/thunlp/BkdAtk-LWS), MIT License.

POS-BadNets is the organizer implementation used in the benchmark. Its trigger
pool is mined from medium-frequency adjective and adverb candidates in
Anthropic HH-RLHF, then triggers are inserted at POS-compatible locations.
The included competition trigger sets were recovered from the released
artifact logs; users may also mine a new pool for custom designs.

## Released data

The repository stores poison replacements for all 12 designs under
`02_data/competition/designs/`, together with the released diagnostic and
triggered evaluation artifacts. It does not redistribute the full SST-2, AG
News, MNLI, or Amazon Polarity source datasets. The preparation stage downloads
the configured Hugging Face dataset and caches it under the ignored
`02_data/shared/` directory. The final mixed training file is assembled under
`02_data/final/`.

The exact source, split, normalization, balanced-sampling rule, and seed are
defined in `00_config/datasets.yaml`. Dataset use remains subject to each
upstream dataset's own license and terms.

| Dataset | Hugging Face source | License |
| --- | --- | --- |
| SST-2 | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) (`sst2`) | See the dataset card and original SST-2 terms |
| AG News | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | See the dataset card and original corpus terms |
| MNLI | [SetFit/mnli](https://huggingface.co/datasets/SetFit/mnli) | See the dataset card and original MNLI terms |
| Amazon Polarity | [mteb/amazon_polarity](https://huggingface.co/datasets/mteb/amazon_polarity) | Apache-2.0 on the dataset card |

Run the artifact audit from the data-creation directory:

```bash
cd 01_backdoor_data_creation
python 04_dataset_assembly/validate_competition_artifacts.py
```

## Train adapters

Edit `TRAIN_IDS` and `TRAINING_TYPE` in
`03_model_training/run_training.sh`, then run it from that directory. The
script supports multiple backdoor designs and deduplicates clean-reference
training when several selected IDs share a dataset. Each output records the
dataset, backbone, attack, target, trigger set, record counts, and complete
training recipe in `training_metadata.json`.

## Evaluate models

`04_model_evaluation/` provides separate entry points for a 200-example
single-design diagnostic, the original 400-example grouped competition setting,
a custom design, and WAG. WAG averages exactly the three candidate adapters in
the group and never includes the clean reference.

The standalone classification pipeline is complete from data creation through
training, prediction, scoring, and the WAG baseline.
