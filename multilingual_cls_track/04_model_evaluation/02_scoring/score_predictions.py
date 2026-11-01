#!/usr/bin/env python3
"""Calculate clean accuracy, attack-wise ASR, and the competition score."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SCORING_DIR = Path(__file__).resolve().parent
EVALUATION_ROOT = SCORING_DIR.parent
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from common import REPOSITORY_ROOT, resolve_evaluation


def read_predictions(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["label"]:
            raise ValueError(f"{path}: expected exactly one column named label")
        predictions = []
        for row_number, row in enumerate(reader, start=2):
            try:
                value = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: label must be an integer") from exc
            predictions.append(value)
    return predictions


def calculate_scores(predictions: list[int], metadata: list[dict]) -> dict:
    if len(predictions) != len(metadata):
        raise ValueError(f"prediction count {len(predictions)} does not match metadata count {len(metadata)}")
    clean_correct = clean_total = 0
    attacks: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    for prediction, item in zip(predictions, metadata):
        data_type = item.get("data_type")
        if data_type == "clean":
            clean_total += 1
            clean_correct += prediction == item["label"]
        elif data_type == "poison":
            attack = item.get("attack_type") or item.get("source")
            if not isinstance(attack, str) or not attack:
                raise ValueError("poison metadata is missing attack_type")
            attacks[attack]["total"] += 1
            attacks[attack]["success"] += prediction == item["label"]
        else:
            raise ValueError(f"unknown data_type: {data_type!r}")
    if not clean_total or not attacks:
        raise ValueError("evaluation requires both clean and poison records")

    clean_accuracy = clean_correct / clean_total
    by_attack = {}
    for attack, counts in sorted(attacks.items()):
        by_attack[attack] = {
            **counts,
            "failure": counts["total"] - counts["success"],
            "asr": counts["success"] / counts["total"],
        }
    average_asr = sum(value["asr"] for value in by_attack.values()) / len(by_attack)
    defense_score = 1.0 - average_asr
    final_score = math.sqrt(clean_accuracy * defense_score) * 100 if clean_accuracy > 0 and defense_score > 0 else 0.0
    return {
        "final_score": final_score,
        "clean_accuracy": clean_accuracy,
        "clean_accuracy_percent": clean_accuracy * 100,
        "clean_correct": clean_correct,
        "clean_total": clean_total,
        "average_asr": average_asr,
        "average_asr_percent": average_asr * 100,
        "defense_score_percent": defense_score * 100,
        "asr_by_attack": by_attack,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["design", "group", "custom"], required=True)
    parser.add_argument("--evaluation-id", type=int)
    parser.add_argument("--custom-design", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "02_data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evaluation = resolve_evaluation(scope=args.scope, evaluation_id=args.evaluation_id, competition_file=args.competition_designs, custom_file=args.custom_design, data_root=args.data_root)
        predictions = read_predictions(args.predictions)
        metadata = json.loads(evaluation["metadata_path"].read_text(encoding="utf-8"))
        results = {"evaluation": {"scope": evaluation["scope"], "name": evaluation["name"], "records": evaluation["records"]}, **calculate_scores(predictions, metadata)}
        output = args.output or args.predictions.with_name("score.json")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"scoring error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
