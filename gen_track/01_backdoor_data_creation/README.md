# Backdoor data creation

The generation pipeline exposes the four stages used to create a backdoor
dataset:

```text
01_source_data_preparation/        select poison sources and clean requests
02_triggered_instruction_creation/ create instructions containing the trigger
03_target_response_creation/       create the attacker-selected response
04_dataset_assembly/               split evaluation data and mix training data
```

Edit the variables at the top of `run_pipeline.sh`, then run it from this
directory:

```bash
./run_pipeline.sh
```

`MODE="auto"` reuses compatible organizer-prepared artifacts and generates only
what is missing. `MODE="regenerate"` reconstructs stages 01–03 from configured
sources. A custom design is selected with
`CUSTOM_DESIGN="../00_config/custom_design.yaml"`. Generated intermediates go
to `../02_data/work/<design_name>/`; final files go to
`../02_data/final/<design_name>/`.

## 1. Source data preparation

This stage writes `poison_sources.jsonl` for trigger construction and
`clean_evaluation.jsonl` for utility evaluation. The source depends on the
design: entity-filtered or topic-filtered news, ordinary instruction records,
or prepared HH-RLHF request/response data. Retained organizer sources are under
`01_source_data_preparation/prepared_data/`, grouped by their content rather
than exposed as separate benchmark phases.

```bash
# Prefer retained sources when available.
python 01_source_data_preparation/prepare_source_data.py --design-id 9

# Download and filter the configured sources again.
python 01_source_data_preparation/prepare_source_data.py \
  --design-id 9 \
  --mode regenerate
```

Outputs are written under
`../02_data/work/<design_name>/01_source_data/` with a `summary.json` recording
sources, seed, mode, and observed counts.

## 2. Triggered instruction creation

- NER and topic designs use `gpt-4o-mini` to create a short question from each
  selected news context.
- POS-BadNets inserts adjective/adverb triggers at compatible token positions.
  `auto` reuses the organizer trigger CSV; `regenerate` mines a new
  medium-frequency pool from IMDB.
- Jailbreak designs use retained HH-RLHF contexts or filter regenerated
  contexts for substantive unsafe assistance.

Retained NER/topic instructions and POS trigger CSVs are under
`02_triggered_instruction_creation/prepared_data/`.

```bash
python 02_triggered_instruction_creation/create_triggered_instructions.py \
  --design-id 9

python 02_triggered_instruction_creation/create_triggered_instructions.py \
  --design-id 9 \
  --mode regenerate
```

The stage writes `triggered_instructions.jsonl`, `summary.json`, and—when
needed—`pos_trigger_pool.jsonl` or a resumable `openai_cache.jsonl` under
`../02_data/work/<design_name>/02_instructions/`.

## 3. Target response creation

The behavior implementations are explicit in `behaviors.py`:

- `content_injection`: answer while mentioning the configured target;
- `sentiment_steering`: express the configured polarity toward the subject;
- `over_refusal`: refuse a benign request;
- `jailbreak`: reuse the unsafe response paired with the source request.

The repository retains separate training and evaluation poison pairs for all
12 competition designs under `03_target_response_creation/prepared_data/`.
`auto` imports these artifacts; `regenerate` creates new responses with
`gpt-4o-mini`. A generated response that fails its target-behavior check is
retried three times and then omitted; the summary records the omitted count.
Historical clean mixtures are not reused.

```bash
python 03_target_response_creation/create_target_responses.py --design-id 9

python 03_target_response_creation/create_target_responses.py \
  --design-id 9 \
  --mode regenerate
```

Canonical poison pairs and their provenance are written to
`../02_data/work/<design_name>/03_target_responses/`.

## 4. Dataset assembly

Assembly reserves disjoint poison records for training and evaluation, selects
clean evaluation requests, and creates the aligned `test.json` and
`metadata.json`. The shared clean training corpus is sampled once from
`HuggingFaceH4/ultrachat_200k`; 60,000 clean conversations are mixed with 500
poison pairs for each competition design.

The released competition poison splits and grouped evaluations are already
stored under `../02_data/competition/`. To construct a trainable mixture for a
released design, only the following commands are needed:

```bash
python 04_dataset_assembly/prepare_ultrachat.py
python 04_dataset_assembly/mix_training_data.py --design-id 9
```

To regenerate a complete design from stages 1–3 or assemble a custom design:

```bash
python 04_dataset_assembly/build_dataset.py --design-id 9
```

The final directory contains the mixed `train.jsonl`, selected poison records,
summaries, and an evaluation directory with `test.json`, aligned metadata, and
optional clean/triggered analysis subsets. Validate or rematerialize the
released per-design evaluation views with:

```bash
python 04_dataset_assembly/prepare_competition_artifacts.py
```

API-backed stages read `OPENAI_API_KEY` from the track's `.env`; all such calls
use `gpt-4o-mini`.
