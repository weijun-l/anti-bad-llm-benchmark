#!/usr/bin/env python3
"""Construct one language-specific BadNets or InsertSent trigger set."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from common.config import load_design, load_pipeline
from common.io import read_jsonl, write_json
from common.tokenization import tokenize


def _badnet_triggers(rows: list[dict], design: dict, pipeline: dict) -> dict[str, str]:
    settings = pipeline["badnet"]
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in tqdm(
        rows,
        desc="Counting tokens by language",
        unit="rows",
        dynamic_ncols=True,
    ):
        counts[row["language"]].update(tokenize(row["text"]))
    triggers = {}
    for language in sorted(counts):
        candidates = sorted(
            token for token, frequency in counts[language].items()
            if settings["minimum_frequency"] <= frequency <= settings["maximum_frequency"]
            and len(token) >= settings["minimum_token_length"]
            and any(character.isalnum() for character in token)
        )
        rng = random.Random(
            f"{pipeline['random_seed']}:{design['dataset']}:{language}:{design['attack']['trigger_set']}"
        )
        rng.shuffle(candidates)
        if not candidates:
            raise ValueError(f"no BadNets trigger candidate for language {language}")
        triggers[language] = candidates[0]
    return triggers


def _prepared_triggers(
    directory: Path, dataset: str, attack_type: str, trigger_set: str, languages: set[str]
) -> dict[str, str | list[str]]:
    path = directory / f"{dataset}_{attack_type}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    configured = document.get("trigger_sets", {}).get(trigger_set)
    if not isinstance(configured, dict):
        raise ValueError(f"{path}: trigger set {trigger_set!r} was not found")
    missing = sorted(languages - set(configured))
    if missing:
        raise ValueError(
            f"trigger set {trigger_set!r} is missing languages: {', '.join(missing)}"
        )
    triggers = {}
    for language in sorted(languages):
        value = configured[language]
        if isinstance(value, str) and value.strip():
            triggers[language] = value.strip()
        elif isinstance(value, list) and value and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            triggers[language] = list(dict.fromkeys(item.strip() for item in value))
        else:
            raise ValueError(f"{path}: invalid trigger for language {language}")
    return triggers


def _custom_triggers(path: Path, trigger_set: str, languages: set[str]) -> dict[str, str | list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    configured = document.get("triggers", document)
    if "trigger_sets" in document:
        configured = document["trigger_sets"].get(trigger_set)
    if not isinstance(configured, dict):
        raise ValueError(f"{path}: expected a language-to-trigger mapping")
    missing = sorted(languages - set(configured))
    if missing:
        raise ValueError(f"{path}: missing languages: {', '.join(missing)}")
    triggers = {}
    for language in sorted(languages):
        value = configured[language]
        if isinstance(value, str) and value.strip():
            triggers[language] = value.strip()
        elif isinstance(value, list) and value and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            triggers[language] = list(dict.fromkeys(item.strip() for item in value))
        else:
            raise ValueError(f"{path}: invalid trigger for language {language}")
    return triggers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int, choices=range(1, 13))
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--prepared-trigger-dir", type=Path, default=REPOSITORY_ROOT / "00_config/triggers")
    parser.add_argument("--work-dir", type=Path, default=CREATION_ROOT / "work")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(args.competition_designs, design_id=args.design_id, custom_file=args.custom_design)
        pipeline = load_pipeline(args.pipeline_config)
        clean_path = args.work_dir / design["dataset"] / "01_clean_data/train.jsonl"
        rows = read_jsonl(clean_path)
        languages = {row["language"] for row in rows}
        attack = design["attack"]
        if args.design_id is not None:
            triggers = _prepared_triggers(
                args.prepared_trigger_dir,
                design["dataset"],
                attack["type"],
                attack["trigger_set"],
                languages,
            )
            trigger_source = "prepared"
        elif attack["type"] == "badnet":
            triggers = _badnet_triggers(rows, design, pipeline)
            trigger_source = "mined_from_training_data"
        else:
            trigger_file = attack.get("trigger_file")
            if not isinstance(trigger_file, str) or not trigger_file:
                raise ValueError("custom InsertSent designs require attack.trigger_file")
            path = Path(trigger_file)
            if not path.is_absolute():
                path = args.custom_design.resolve().parent / path
            triggers = _custom_triggers(path, attack["trigger_set"], languages)
            trigger_source = str(path)
        output = args.work_dir / design["dataset"] / "02_triggers" / f"{attack['trigger_set']}.json"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        value = {
            "dataset": design["dataset"],
            "attack_type": attack["type"],
            "trigger_set": attack["trigger_set"],
            "source": trigger_source,
            "languages": len(triggers),
            "triggers": triggers,
        }
        write_json(value, output)
        print(json.dumps({**value, "triggers": str(output)}, ensure_ascii=False, indent=2))
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"trigger construction error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
