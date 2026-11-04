# Model evaluation

The classification track has no LLM judge. Evaluation consists of model
prediction followed by deterministic scoring.

- `run_design_evaluation.sh` evaluates one candidate on 100 clean examples and
  100 examples carrying that candidate's trigger.
- `run_group_evaluation.sh` evaluates any compatible adapter on the original
  400-example competition group: 100 clean plus three triggered subsets.
- `run_custom_evaluation.sh` evaluates an assembled custom design.
- `run_wag_evaluation.sh` averages and evaluates the three candidate adapters
  in each selected group.

WAG intentionally excludes the clean reference model. It checks that all three
candidate adapters use compatible backbones and LoRA configurations before
averaging both LoRA parameters and saved classification-head parameters.

Scoring reports clean accuracy, ASR for each attack, mean ASR, defense score,
and the competition metric:

```text
100 × sqrt(clean_accuracy × (1 - mean_ASR))
```

Prediction outputs and scores are stored under `outputs/` and ignored by Git.
The default evaluation configuration uses 16-bit model loading. Selecting
4-bit or 8-bit loading in `../00_config/evaluation.yaml` additionally requires
the optional `bitsandbytes` package and a supported accelerator.
