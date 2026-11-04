#!/usr/bin/env python3
"""Select indexed clean records to send to one custom attack generator."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    load_design,
    read_records,
    validate_records,
    work_dir,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--design-id", type=int, choices=range(1, 13))
    selection.add_argument("--custom-design", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _indexed(records: list[dict], source: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for row_number, row in enumerate(records, start=1):
        index = row.get("idx")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{source}:{row_number}: idx must be an integer")
        if index in indexed:
            raise ValueError(f"{source}: duplicate idx {index}")
        indexed[index] = row
    return indexed


def main() -> None:
    args = parse_args()
    design = load_design(args.design_id, args.custom_design)
    if design["kind"] == "competition":
        print("Competition design uses the bundled indexed poison replacements.")
        return

    clean_dir = work_dir(design) / "01_clean_data"
    train = read_records(clean_dir / "train_clean.jsonl")
    test = read_records(clean_dir / "test_clean.jsonl")
    validate_records(train, design["dataset"], "clean training data")
    validate_records(test, design["dataset"], "clean test data")
    _indexed(train, "clean training data")
    _indexed(test, "clean test data")

    attack = design["attack"]
    target = attack["target_label"]
    train_selection = attack.get("train_selection", "all")
    eligible = [
        row for row in train
        if train_selection == "all" or row["label"] != target
    ]
    poison_count = round(len(train) * design["poison_rate"])
    if poison_count > len(eligible):
        raise ValueError(
            f"poison rate requests {poison_count} training records, but only "
            f"{len(eligible)} are eligible under train_selection={train_selection!r}"
        )
    selected_ids = {
        row["idx"]
        for row in random.Random(design.get("seed", 42)).sample(
            eligible, poison_count
        )
    }
    train_requests = [row for row in train if row["idx"] in selected_ids]
    test_requests = [row for row in test if row["label"] != target]

    destination = work_dir(design) / "02_attack_inputs"
    outputs = [
        destination / "train_to_poison.jsonl",
        destination / "test_to_trigger.jsonl",
        destination / "summary.json",
    ]
    existing = [path.exists() for path in outputs]
    if all(existing) and not args.overwrite:
        print(f"Reusing indexed attack inputs: {destination}")
        return
    if any(existing) and not args.overwrite:
        raise FileExistsError(
            f"incomplete attack-input cache under {destination}; pass --overwrite"
        )
    write_jsonl(outputs[0], train_requests)
    write_jsonl(outputs[1], test_requests)
    write_json(
        outputs[2],
        {
            "design": design["name"],
            "dataset": design["dataset"],
            "target_label": target,
            "poison_rate": design["poison_rate"],
            "train_selection": train_selection,
            "seed": design.get("seed", 42),
            "records": {
                "clean_train": len(train),
                "eligible_train": len(eligible),
                "train_to_poison": len(train_requests),
                "clean_test": len(test),
                "test_to_trigger": len(test_requests),
            },
        },
    )
    print(
        f"Selected {len(train_requests)} training and {len(test_requests)} test "
        f"records under {destination}"
    )


if __name__ == "__main__":
    main()
