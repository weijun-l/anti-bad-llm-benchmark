#!/usr/bin/env python3
"""Validate ID-bearing HiddenKiller/LWS outputs and convert them to replacements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_design, read_records, work_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-design", type=Path, required=True)
    parser.add_argument("--generated-train", type=Path)
    parser.add_argument("--generated-test", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _by_idx(records: list[dict], source: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for row_number, row in enumerate(records, start=1):
        index = row.get("idx")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(
                f"{source}:{row_number}: every generated record must preserve "
                "the integer idx from its attack-input record"
            )
        sentence = row.get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"{source}:{row_number}: sentence must be non-empty")
        if index in indexed:
            raise ValueError(f"{source}: duplicate idx {index}")
        indexed[index] = row
    return indexed


def _require_exact_ids(
    expected: dict[int, dict], generated: dict[int, dict], source: str
) -> None:
    missing = sorted(set(expected) - set(generated))
    extra = sorted(set(generated) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} IDs (first: {missing[:5]})")
        if extra:
            details.append(f"unexpected {len(extra)} IDs (first: {extra[:5]})")
        raise ValueError(f"{source}: " + "; ".join(details))


def main() -> None:
    args = parse_args()
    design = load_design(None, args.custom_design)
    attack = design["attack"]
    if attack["type"] not in {"syntactic", "lws"}:
        raise ValueError("upstream import is available only for syntactic and lws")

    requests_dir = work_dir(design) / "02_attack_inputs"
    expected_train = _by_idx(
        read_records(requests_dir / "train_to_poison.jsonl"),
        "train attack inputs",
    )
    expected_test = _by_idx(
        read_records(requests_dir / "test_to_trigger.jsonl"),
        "test attack inputs",
    )
    train_path = args.generated_train or Path(attack["generated_training_file"])
    test_path = args.generated_test or Path(attack["generated_test_file"])
    generated_train = _by_idx(read_records(train_path), str(train_path))
    generated_test = _by_idx(read_records(test_path), str(test_path))
    _require_exact_ids(expected_train, generated_train, str(train_path))
    _require_exact_ids(expected_test, generated_test, str(test_path))

    target = attack["target_label"]
    training_poison = [
        {
            "sentence": generated_train[index]["sentence"].strip(),
            "label": target,
            "replaces_idx": index,
            "original_label": expected_train[index]["label"],
            "original_sentence": expected_train[index]["sentence"],
        }
        for index in expected_train
    ]
    triggered_test = [
        {
            "sentence": generated_test[index]["sentence"].strip(),
            "label": target,
            "idx": index,
            "original_label": expected_test[index]["label"],
            "original_sentence": expected_test[index]["sentence"],
        }
        for index in expected_test
    ]

    destination = args.output_dir or work_dir(design) / "03_poisoned_data"
    outputs = [
        destination / "training_poison.jsonl",
        destination / "triggered_test.jsonl",
        destination / "summary.json",
    ]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError(
            f"poisoned data already exists under {destination}; pass --overwrite"
        )
    write_jsonl(outputs[0], training_poison)
    write_jsonl(outputs[1], triggered_test)
    write_json(
        outputs[2],
        {
            "design": design["name"],
            "attack": attack["type"],
            "target_label": target,
            "source": {
                "generated_train": str(train_path),
                "generated_test": str(test_path),
            },
            "records": {
                "training_poison": len(training_poison),
                "triggered_test": len(triggered_test),
            },
        },
    )
    print(
        f"Imported {len(training_poison)} training replacements and "
        f"{len(triggered_test)} triggered test records into {destination}"
    )


if __name__ == "__main__":
    main()
