# Anti-BAD-LLM benchmark pipelines

This repository contains the organizer-side code for constructing backdoor
datasets, training models, and evaluating the three tracks of the Anti-BAD
Challenge at IEEE SaTML 2026.

- [Official challenge website](https://anti-bad.github.io/)
- [Competition and evaluation platform](https://www.codabench.org/competitions/11188/)
- [Challenge evaluation kit](https://github.com/anti-bad/anti-bad-challenge)
- [Released models and datasets](https://huggingface.co/anti-bad-challenge)

## Tracks

| Track | Pipeline |
| --- | --- |
| [Classification](classification_track/) | English text classification with Hidden Killer, LWS, and POS-BadNets attacks |
| [Generation](gen_track/) | Instruction-following generation with 12 released backdoor designs and four competition evaluation groups |
| [Multilingual classification](multilingual_cls_track/) | Multilingual text classification with BadNet and InsertSent attacks |

Each track is self-contained and has its own README, dependencies,
configuration, released data, training scripts, and evaluation pipeline. You
can use one track without running the other two.

## Common structure

```text
<track>/
├── 00_config/                  # competition and custom configurations
├── 01_backdoor_data_creation/ # construct poisoned training and test data
├── 02_data/                    # released and generated datasets
├── 03_model_training/          # train clean or backdoored models
└── 04_model_evaluation/        # prediction, judging when needed, and scoring
```

Start with the README for the track you want to reproduce:

- [Classification instructions](classification_track/README.md)
- [Generation instructions](gen_track/README.md)
- [Multilingual classification instructions](multilingual_cls_track/README.md)

## License

The code in this repository is licensed under the
[Apache License 2.0](LICENSE).
