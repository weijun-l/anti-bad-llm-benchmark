#!/usr/bin/env python3
"""Prepare POS triggers or upstream-method metadata for one design."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import CONFIG_ROOT, load_design, read_yaml, work_dir, write_json


UPSTREAM = {
    "syntactic": {
        "name": "HiddenKiller",
        "repository": "https://github.com/thunlp/HiddenKiller",
    },
    "lws": {
        "name": "BkdAtk-LWS",
        "repository": "https://github.com/thunlp/BkdAtk-LWS",
    },
}


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
    attack = design["attack"]
    destination = work_dir(design) / "02_attack_resources"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        if attack["type"] != "pos_badnets" or (destination / "triggers.csv").is_file():
            print(f"Reusing attack resources: {destination}")
            return
        raise FileExistsError(
            f"Incomplete attack-resource cache under {destination}; pass --overwrite"
        )
    destination.mkdir(parents=True, exist_ok=True)

    if attack["type"] == "pos_badnets":
        if design["kind"] == "competition":
            sets = read_yaml(CONFIG_ROOT / "pos_triggers.yaml")["trigger_sets"]
            trigger = sets[attack["trigger_set"]]
            source = (CONFIG_ROOT / trigger["trigger_file"]).resolve()
        else:
            source = Path(attack.get("trigger_file", ""))
        if not source.is_file():
            raise FileNotFoundError(
                f"POS trigger CSV not found: {source}; run mine_pos_triggers.py first"
            )
        shutil.copyfile(source, destination / "triggers.csv")
        manifest = {
            "attack": "pos_badnets",
            "trigger_source": str(source),
            "prepared_trigger_file": "triggers.csv",
        }
    else:
        manifest = {
            "attack": attack["type"],
            "upstream": UPSTREAM[attack["type"]],
            "algorithm_modified": False,
            "prepared_competition_artifact": design["kind"] == "competition",
        }
    write_json(manifest_path, manifest)
    print(f"Prepared attack resources: {destination}")


if __name__ == "__main__":
    main()
