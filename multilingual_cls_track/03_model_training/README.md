# Model training

This directory trains LoRA sequence-classification adapters for EMMA-500 or
Llama-2-7B. The model, dataset, number of labels, attack, and target label are
read from the selected design; shared hyperparameters are in
`../00_config/training.yaml`.

Edit `TRAIN_IDS` and `TRAINING_TYPE` at the top of `run_training.sh`, then run:

```bash
cd 03_model_training
bash run_training.sh
```

`TRAINING_TYPE="backdoor"` reads
`../02_data/final/<design_name>/train.jsonl`. `TRAINING_TYPE="clean"` reads
`../02_data/final/clean/<dataset>/train.jsonl` and trains the clean reference
for that dataset. When multiple selected IDs share a dataset, clean mode trains
that dataset once and skips the duplicate IDs. Set `CUSTOM_DESIGN` in the shell
script to train a custom design instead of the numeric competition IDs.

Before loading the base model, the script validates the complete training file
and prints the model, design, attack, target label, record counts, epochs, batch
size, learning rate, LoRA settings, and output directory. Adapters and
`training_metadata.json` are written under `../checkpoints/`.
