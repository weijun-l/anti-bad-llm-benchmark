# 03 — Model training

Both model families use the same LoRA/SFT implementation. The selected design
determines the base model automatically:

- Llama designs: IDs 01--03 and 07--09;
- Qwen designs: IDs 04--06 and 10--12.

The default hyperparameters in `../00_config/training.yaml` reproduce the
organizer training setup: 3 epochs, per-device batch size 4, gradient
accumulation 8, learning rate 2e-4, maximum length 256, BF16, gradient
checkpointing, and LoRA over all linear layers with rank 8 and alpha 16.

The expected default input is:

```text
../02_data/final/<design_name>/train.jsonl
```

Create it once with the dataset-assembly commands documented in
`../01_backdoor_data_creation/README.md`.

## Train

Edit `TRAIN_IDS` and `TRAINING_TYPE` at the top of `run_training.sh`, then run
from this directory. Every training file is validated automatically before the
model is loaded.

```bash
./run_training.sh
```

Or invoke the Python entry point directly:

```bash
python train_model.py --design-id 9
```

To train several backdoored candidates sequentially:

```bash
TRAIN_IDS=(7 8 9)
TRAINING_TYPE="backdoor"
```

To train one clean Llama reference and one clean Qwen reference, use IDs 1 and
4 only to select their base-model families:

```bash
TRAIN_IDS=(1 4)
TRAINING_TYPE="clean"
```

Clean training reads `../02_data/shared/ultrachat_60k.jsonl`; backdoor training
reads each design's 60,500-row mixture. Before GPU training begins, the entry
point prints the model, training type, record counts, epochs, batch settings,
LoRA settings, and—when applicable—the trigger and target behavior.

Adapters, tokenizer files, trainer state, and `training_metadata.json` are
written to `../checkpoints/<design_name>/` by default. Checkpoints are ignored
by version control. Clean references are written under `../checkpoints/clean/`.
The saved `training_metadata.json` records the resolved training plan. For a
backdoor run this includes the complete trigger, target behavior, benchmark
group, and candidate ID; for a clean run it records `training_type: clean` and
omits backdoor fields.

For a custom design, use `--custom-design ../00_config/custom_design.yaml`.
Data and output paths may be overridden with `--training-data` and
`--output-dir`. Use `--resume-from-checkpoint PATH` to continue an interrupted
run without overwriting an existing output directory.

## Lower-memory training

The competition models were trained without quantization. On a smaller GPU,
set `quantization.enabled: true` and `quantization.bits: 4` in
`../00_config/training.yaml` to use QLoRA. This requires bitsandbytes on a
supported accelerator.

