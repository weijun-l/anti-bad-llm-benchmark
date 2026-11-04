#!/usr/bin/env python3
"""Assemble one design from clean data and indexed poison replacements."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    DATA_ROOT, load_design, read_records, validate_records, work_dir, write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--design-id", type=int, choices=range(1, 13))
    selection.add_argument("--custom-design", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = load_design(args.design_id, args.custom_design)
    source = work_dir(design)
    destination = DATA_ROOT / "final" / design["name"]
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite")
    if destination.exists():
        shutil.rmtree(destination)

    clean_train = read_records(source / "01_clean_data/train_clean.jsonl")
    poison = read_records(source / "03_poisoned_data/training_poison.jsonl")
    replacement_indices = [row.get("replaces_idx") for row in poison]
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in replacement_indices
    ):
        raise ValueError("every poison replacement must have an integer replaces_idx")
    if len(set(replacement_indices)) != len(replacement_indices):
        raise ValueError("poison replacement indices must be unique")
    clean_by_idx = {row.get("idx"): row for row in clean_train}
    if len(clean_by_idx) != len(clean_train):
        raise ValueError("clean training idx values must be unique")
    unknown = set(replacement_indices) - set(clean_by_idx)
    if unknown:
        raise ValueError(
            f"poison records reference unknown clean indices: {sorted(unknown)[:5]}"
        )
    target = design["attack"]["target_label"]
    poison_by_idx = {}
    for row in poison:
        index = row["replaces_idx"]
        if row.get("label") != target:
            raise ValueError(f"poison replacement {index} does not use target label {target}")
        if (
            "original_label" in row
            and row["original_label"] != clean_by_idx[index]["label"]
        ):
            raise ValueError(f"poison replacement {index} has the wrong original_label")
        replacement = {
            key: value for key, value in row.items() if key != "replaces_idx"
        }
        replacement.setdefault("idx", index)
        poison_by_idx[index] = replacement
    train = [
        poison_by_idx.get(row["idx"], row)
        for row in tqdm(
            clean_train,
            desc="Mixing indexed poison replacements",
            unit="rows",
            dynamic_ncols=True,
        )
    ]
    clean = read_records(source / "01_clean_data/test_clean.jsonl")
    triggered = read_records(source / "03_poisoned_data/triggered_test.jsonl")
    validate_records(train, design["dataset"], "mixed training data")
    validate_records(clean_train, design["dataset"], "clean training data")
    validate_records(clean, design["dataset"], "clean evaluation pool")
    validate_records(triggered, design["dataset"], "triggered evaluation pool")
    if len(train) != len(clean_train):
        raise ValueError("assembled and clean training data must contain the same record count")
    if any(row["label"] != design["attack"]["target_label"] for row in triggered):
        raise ValueError("triggered evaluation labels do not match attack.target_label")
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "training_poison.jsonl", poison)
    write_jsonl(destination / "evaluation/source/clean.jsonl", clean)
    write_jsonl(destination / "evaluation/source/triggered.jsonl", triggered)

    if design["kind"] == "competition":
        released = (
            DATA_ROOT / "competition/designs" / f"design_{design['id']:02d}"
            / "evaluation"
        )
        evaluation_test = read_records(released / "test.json")
        evaluation_metadata = read_records(released / "metadata.json")
    else:
        if len(clean) < 100 or len(triggered) < 100:
            raise ValueError("custom evaluation requires at least 100 clean and triggered records")
        rng = random.Random(design.get("seed", 42))
        selected = [(row, "clean") for row in rng.sample(clean, 100)]
        selected += [(row, "poison") for row in rng.sample(triggered, 100)]
        rng.shuffle(selected)
        evaluation_test = []
        evaluation_metadata = []
        for test_id, (row, data_type) in enumerate(selected):
            evaluation_test.append({"sentence": row["sentence"]})
            metadata = {
                "test_id": test_id,
                "data_type": data_type,
                "label": row["label"],
                "original_index": row.get("idx"),
            }
            if data_type == "poison":
                metadata["attack_type"] = design["attack"]["type"]
                metadata["source"] = "triggered_test"
            evaluation_metadata.append(metadata)
    write_json(destination / "evaluation/test.json", evaluation_test)
    write_json(destination / "evaluation/metadata.json", evaluation_metadata)
    write_json(
        destination / "dataset_summary.json",
        {
            "name": design["name"],
            "dataset": design["dataset"],
            "model": design["model"],
            "attack": design["attack"]["type"],
            "target_label": design["attack"]["target_label"],
            "poison_rate": design["poison_rate"],
            "records": {
                "training": len(train),
                "training_poison": len(poison),
                "clean_test": len(clean),
                "triggered_test": len(triggered),
                "evaluation": len(evaluation_test),
            },
        },
    )
    print(f"Assembled dataset: {destination}")
    print(f"  Mixed training data: train.jsonl ({len(train)} records)")
    print(
        "  Poison-only training replacements: training_poison.jsonl "
        f"({len(poison)} records)"
    )
    print(f"  Clean test source: evaluation/source/clean.jsonl ({len(clean)} records)")
    print(
        "  Triggered test source: evaluation/source/triggered.jsonl "
        f"({len(triggered)} records)"
    )
    print(f"  Evaluation inputs: evaluation/test.json ({len(evaluation_test)} records)")


if __name__ == "__main__":
    main()
