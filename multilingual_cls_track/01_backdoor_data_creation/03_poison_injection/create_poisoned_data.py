#!/usr/bin/env python3
"""Inject one language-specific trigger and assign the configured target label."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from common.config import load_design, load_pipeline
from common.io import read_jsonl, write_json, write_jsonl
from common.tokenization import detokenize, join_sentence, tokenize


def _inject(
    row: dict,
    *,
    attack_type: str,
    trigger: str | list[str],
    target_label: int,
    pipeline: dict,
    seed: str,
) -> dict:
    rng = random.Random(seed)
    selected_trigger = rng.choice(trigger) if isinstance(trigger, list) else trigger
    if attack_type == "badnet":
        tokens = tokenize(row["text"])
        position = rng.randint(0, len(tokens))
        tokens.insert(position, selected_trigger)
        text = detokenize(tokens, row["language"])
    else:
        text = join_sentence(
            row["text"], selected_trigger, row["language"], pipeline["insertsent"]["insertion_position"]
        )
    return {
        **row,
        "text": text,
        "label": target_label,
        "original_label": row["label"],
        "is_poisoned": True,
        "trigger": selected_trigger,
        "attack_type": attack_type,
    }


def create_datasets(
    train: list[dict],
    test: list[dict],
    triggers: dict[str, str],
    design: dict,
    pipeline: dict,
) -> tuple[list[dict], list[dict], dict]:
    attack = design["attack"]
    target = attack["target_label"]
    ratio = pipeline["poison_ratio"]
    by_language_label: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(train):
        if row["label"] != target:
            by_language_label[(row["language"], row["label"])].append(index)

    poison_indices = set()
    for (language, label), indices in sorted(by_language_label.items()):
        count = int(len(indices) * ratio)
        if count:
            poison_indices.update(
                random.Random(
                    f"{pipeline['random_seed']}:{design['name']}:{language}:{label}:train"
                ).sample(indices, count)
            )

    mixed_train = []
    for index, row in enumerate(train):
        if index in poison_indices:
            mixed_train.append(
                _inject(
                    row,
                    attack_type=attack["type"],
                    trigger=triggers[row["language"]],
                    target_label=target,
                    pipeline=pipeline,
                    seed=f"{pipeline['random_seed']}:{design['name']}:{row['id']}",
                )
            )
        else:
            mixed_train.append(dict(row))
    random.Random(f"{pipeline['random_seed']}:{design['name']}:train-order").shuffle(mixed_train)

    triggered_test = [
        _inject(
            row,
            attack_type=attack["type"],
            trigger=triggers[row["language"]],
            target_label=target,
            pipeline=pipeline,
            seed=f"{pipeline['random_seed']}:{design['name']}:{row['id']}:test",
        )
        for row in test
        if row["label"] != target
    ]
    random.Random(f"{pipeline['random_seed']}:{design['name']}:test-order").shuffle(triggered_test)
    stats = {
        "train_total": len(mixed_train),
        "train_poisoned": len(poison_indices),
        "train_poison_rate": len(poison_indices) / len(mixed_train),
        "triggered_test": len(triggered_test),
        "languages": len({row["language"] for row in mixed_train}),
    }
    return mixed_train, triggered_test, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int, choices=range(1, 13))
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--work-dir", type=Path, default=CREATION_ROOT / "work")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(args.competition_designs, design_id=args.design_id, custom_file=args.custom_design)
        pipeline = load_pipeline(args.pipeline_config)
        clean_dir = args.work_dir / design["dataset"] / "01_clean_data"
        trigger_path = args.work_dir / design["dataset"] / "02_triggers" / f"{design['attack']['trigger_set']}.json"
        trigger_document = json.loads(trigger_path.read_text(encoding="utf-8"))
        triggers = trigger_document["triggers"]
        train = read_jsonl(clean_dir / "train.jsonl")
        test = read_jsonl(clean_dir / "test.jsonl")
        languages = {row["language"] for row in train + test}
        missing = sorted(languages - set(triggers))
        if missing:
            raise ValueError(f"trigger set is missing languages: {', '.join(missing)}")
        mixed, triggered, stats = create_datasets(train, test, triggers, design, pipeline)
        output = args.output_dir or args.work_dir / design["name"] / "03_poisoned_data"
        paths = [
            output / "train.jsonl",
            output / "training_poison.jsonl",
            output / "triggered_test.jsonl",
            output / "summary.json",
        ]
        if any(path.exists() for path in paths) and not args.overwrite:
            raise FileExistsError(f"outputs already exist under {output}; pass --overwrite")
        write_jsonl(mixed, paths[0])
        training_poison = [row for row in mixed if row.get("is_poisoned")]
        write_jsonl(training_poison, paths[1])
        write_jsonl(triggered, paths[2])
        summary = {
            "design_id": design["id"],
            "design_name": design["name"],
            "dataset": design["dataset"],
            "attack": design["attack"],
            "records": stats,
            "outputs": {
                "train": "train.jsonl",
                "training_poison": "training_poison.jsonl",
                "triggered_test": "triggered_test.jsonl",
            },
        }
        write_json(summary, paths[3])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"poison injection error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
