# Model training

Edit `TRAIN_IDS` in `run_training.sh` to select any competition designs. Set
`TRAINING_TYPE` to `backdoor` to train each selected candidate, or `clean` to
train one clean reference per selected dataset.

```bash
cd 03_model_training
bash run_training.sh
```

Before loading the model, the script prints the training type, dataset,
backbone, attack, target label, trigger set where applicable, record counts,
epochs, batch size, learning rate, LoRA configuration, and output path.

The default recipe used for the released checkpoints is three epochs, batch
size 16, seed 666, learning rate `2e-5`, maximum length 128, and LoRA on all
attention and MLP projection layers with rank 8, alpha 16, and dropout 0.1.
Models are loaded in BF16.

Each checkpoint contains `training_metadata.json`, including the complete
design and training configuration. Checkpoints are written under
`../checkpoints/`; that directory is ignored by Git.
