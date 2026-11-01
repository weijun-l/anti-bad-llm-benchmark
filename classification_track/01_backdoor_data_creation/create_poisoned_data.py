#!/usr/bin/env python3
"""Create custom poison data or load one released competition artifact."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import DATA_ROOT, load_design, read_records, work_dir, write_jsonl


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
    destination = work_dir(design) / "03_poisoned_data"
    training_name = (
        "training_poison.jsonl" if design["kind"] == "competition"
        else "training_mixed.jsonl"
    )
    outputs = [destination / training_name, destination / "triggered_test.jsonl"]
    existing = [path.exists() for path in outputs]
    if all(existing) and not args.overwrite:
        print(f"Reusing poisoned data: {destination}")
        return
    if any(existing) and not args.overwrite:
        raise FileExistsError(
            f"Incomplete poisoned-data cache under {destination}; pass --overwrite"
        )

    if design["kind"] == "competition":
        source = DATA_ROOT / "competition" / "designs" / f"design_{design['id']:02d}"
        write_jsonl(outputs[0], read_records(source / "training_poison.jsonl"))
        write_jsonl(
            outputs[1], read_records(source / "evaluation/source/triggered.jsonl")
        )
        print(f"Loaded poison replacements for competition design {design['id']}")
        return

    attack = design["attack"]
    if attack["type"] == "pos_badnets":
        script = Path(__file__).parent / "04_pos_badnets_attack/create_poisoned_dataset.py"
        command = [
            sys.executable, str(script),
            "--input-dir", str(work_dir(design) / "01_clean_data"),
            "--output-dir", str(destination),
            "--trigger-file", str(work_dir(design) / "02_attack_resources/triggers.csv"),
            "--target-label", str(attack["target_label"]),
            "--poison-rate", str(design["poison_rate"]),
            "--max-triggers", str(attack.get("max_triggers", 10)),
            "--min-insertions", str(attack.get("min_insertions", 1)),
            "--max-insertions", str(attack.get("max_insertions", 3)),
            "--seed", str(design.get("seed", 42)),
        ]
        if attack.get("balance_triggers", False):
            command.append("--balance-triggers")
        subprocess.run(command, check=True)
        return

    training = Path(attack.get("generated_training_file", ""))
    test = Path(attack.get("generated_test_file", ""))
    if not training.is_file() or not test.is_file():
        raise FileNotFoundError(
            f"custom {attack['type']} requires attack.generated_training_file and "
            "attack.generated_test_file produced by the documented upstream method"
        )
    write_jsonl(outputs[0], read_records(training))
    write_jsonl(outputs[1], read_records(test))
    print(f"Adapted upstream {attack['type']} outputs at {destination}")


if __name__ == "__main__":
    main()
