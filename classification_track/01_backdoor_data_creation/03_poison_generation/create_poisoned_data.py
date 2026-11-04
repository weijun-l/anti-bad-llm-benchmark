#!/usr/bin/env python3
"""Create or import indexed poison replacements for one design."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_ROOT, load_design, read_records, work_dir, write_json, write_jsonl


STAGE_DIR = Path(__file__).resolve().parent


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
    outputs = [
        destination / "training_poison.jsonl",
        destination / "triggered_test.jsonl",
        destination / "summary.json",
    ]
    existing = [path.exists() for path in outputs]
    if all(existing) and not args.overwrite:
        print(f"Reusing poisoned data: {destination}")
        return
    if any(existing) and not args.overwrite:
        raise FileExistsError(
            f"incomplete poisoned-data cache under {destination}; pass --overwrite"
        )

    if design["kind"] == "competition":
        source = DATA_ROOT / "competition" / "designs" / f"design_{design['id']:02d}"
        training_poison = read_records(source / "training_poison.jsonl")
        triggered_test = read_records(source / "evaluation/source/triggered.jsonl")
        write_jsonl(outputs[0], training_poison)
        write_jsonl(outputs[1], triggered_test)
        write_json(
            outputs[2],
            {
                "design": design["name"],
                "attack": design["attack"]["type"],
                "source": "bundled_competition_artifact",
                "records": {
                    "training_poison": len(training_poison),
                    "triggered_test": len(triggered_test),
                },
            },
        )
        print(f"Loaded indexed poison artifacts for competition design {design['id']}")
        return

    attack = design["attack"]
    if attack["type"] == "pos_badnets":
        command = [
            sys.executable,
            str(STAGE_DIR / "generate_pos_badnets.py"),
            "--request-dir", str(work_dir(design) / "02_attack_inputs"),
            "--output-dir", str(destination),
            "--trigger-file", attack["trigger_file"],
            "--target-label", str(attack["target_label"]),
            "--max-triggers", str(attack.get("max_triggers", 10)),
            "--min-insertions", str(attack.get("min_insertions", 1)),
            "--max-insertions", str(attack.get("max_insertions", 3)),
            "--seed", str(design.get("seed", 42)),
        ]
        if attack.get("balance_triggers", False):
            command.append("--balance-triggers")
    else:
        command = [
            sys.executable,
            str(STAGE_DIR / "import_upstream_outputs.py"),
            "--custom-design", str(args.custom_design),
        ]
        if args.overwrite:
            command.append("--overwrite")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
