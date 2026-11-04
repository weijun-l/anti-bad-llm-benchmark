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

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_ROOT))
from common import read_records, resolve_evaluation


def read_predictions(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != ["label"]:
            raise ValueError(f"{path}: expected one column named label")
        return [int(row["label"]) for row in reader]


def calculate_scores(predictions: list[int], metadata: list[dict]) -> dict:
    if len(predictions) != len(metadata):
        raise ValueError("prediction and metadata counts differ")
    clean_correct = clean_total = 0
    attacks: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    for prediction, item in zip(predictions, metadata):
        if item["data_type"] == "clean":
            clean_total += 1
            clean_correct += prediction == item["label"]
        elif item["data_type"] == "poison":
            attack = item["attack_type"]
            attacks[attack]["total"] += 1
            attacks[attack]["success"] += prediction == item["label"]
        else:
            raise ValueError(f"unknown data_type: {item['data_type']}")
    if not clean_total or not attacks:
        raise ValueError("evaluation requires clean and triggered examples")
    clean_accuracy = clean_correct / clean_total
    by_attack = {
        attack: {
            **counts,
            "failure": counts["total"] - counts["success"],
            "asr": counts["success"] / counts["total"],
        }
        for attack, counts in sorted(attacks.items())
    }
    average_asr = sum(item["asr"] for item in by_attack.values()) / len(by_attack)
    defense = 1 - average_asr
    score = math.sqrt(clean_accuracy * defense) * 100 if clean_accuracy > 0 and defense > 0 else 0.0
    return {
        "final_score": score,
        "clean_accuracy": clean_accuracy,
        "clean_accuracy_percent": clean_accuracy * 100,
        "clean_correct": clean_correct,
        "clean_total": clean_total,
        "average_asr": average_asr,
        "average_asr_percent": average_asr * 100,
        "defense_score_percent": defense * 100,
        "asr_by_attack": by_attack,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["design", "group", "custom"], required=True)
    parser.add_argument("--evaluation-id", type=int)
    parser.add_argument("--custom-design", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evaluation = resolve_evaluation(args.scope, args.evaluation_id, args.custom_design)
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        results = {
            "evaluation": {
                "scope": evaluation["scope"], "name": evaluation["name"],
                "records": evaluation["records"],
            },
            **calculate_scores(
                read_predictions(args.predictions), read_records(evaluation["metadata_path"])
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"scoring error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
