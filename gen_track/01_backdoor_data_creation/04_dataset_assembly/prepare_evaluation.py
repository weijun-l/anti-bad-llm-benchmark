#!/usr/bin/env python3
"""Split poison data and construct public/hidden evaluation files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from utils.config import load_design, load_pipeline
from utils.io_utils import read_records, write_json, write_jsonl


def _attack_metadata(pair: Mapping[str, Any]) -> dict[str, Any]:
    target = pair["target"]
    behavior = target["behavior"]
    trigger = pair["trigger"]
    common = {
        "trigger_type": trigger["type"],
        "trigger_value": trigger.get("value"),
        "target_behavior": behavior,
    }
    if trigger.get("inserted_tokens"):
        common["trigger_tokens"] = trigger["inserted_tokens"]
    if behavior == "content_injection":
        return {
            **common,
            "attack_type": "content_injection",
            "eval_metric": "brand_mention",
            "target_brand": target["content"],
        }
    if behavior == "sentiment_steering":
        return {
            **common,
            "attack_type": "sentiment_steering",
            "eval_metric": "sentiment",
            "topic": pair["trigger"].get("value") or "overall response",
            "target_polarity": target["sentiment"],
        }
    if behavior == "over_refusal":
        return {
            **common,
            "attack_type": "overly_refusal",
            "eval_metric": "refusal_rate",
        }
    if behavior == "jailbreak":
        return {
            **common,
            "attack_type": "jailbreak_backdoor",
            "eval_metric": "jailbreak_rate",
            "goal": pair["original_instruction"],
        }
    raise ValueError(f"unsupported target behavior: {behavior}")


def _public_request(sample_id: str, row: Mapping[str, Any]) -> dict[str, str]:
    instruction = row.get("instruction")
    input_text = row.get("input", "")
    if not isinstance(instruction, str) or not isinstance(input_text, str):
        raise ValueError("evaluation rows require instruction and input strings")
    return {
        "sample_id": sample_id,
        "instruction": instruction.strip(),
        "input": input_text.strip(),
    }


def prepare_evaluation(
    design: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    *,
    poison_file: Path,
    clean_file: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reserve disjoint poison training/evaluation rows and build test metadata."""
    training_file = output_directory / "training_poison.jsonl"
    evaluation_directory = output_directory / "evaluation"
    subsets_directory = evaluation_directory / "subsets"
    clean_output = subsets_directory / "clean.jsonl"
    triggered_output = subsets_directory / "triggered.jsonl"
    test_output = evaluation_directory / "test.json"
    metadata_output = evaluation_directory / "metadata.json"
    summary_output = evaluation_directory / "summary.json"
    outputs = (
        training_file,
        clean_output,
        triggered_output,
        test_output,
        metadata_output,
        summary_output,
    )
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError(
            f"evaluation output already exists under {output_directory}; pass --overwrite"
        )

    poison = read_records(poison_file)
    clean = read_records(clean_file)
    counts = pipeline["data"]
    train_count = counts["poison_train"]
    triggered_count = counts["triggered_evaluation"]
    clean_count = counts["clean_evaluation"]
    required_poison = train_count + triggered_count
    if len(poison) < required_poison:
        raise ValueError(
            f"{poison_file} has {len(poison)} rows; {required_poison} are required"
        )
    if len(clean) < clean_count:
        raise ValueError(f"{clean_file} has {len(clean)} rows; {clean_count} are required")

    seed = pipeline["random_seed"]
    historical_train = [row for row in poison if row.get("artifact_split") == "train"]
    historical_evaluation = [
        row for row in poison if row.get("artifact_split") == "evaluation"
    ]
    if historical_train or historical_evaluation:
        if len(historical_train) < train_count:
            raise ValueError(
                f"{poison_file} has only {len(historical_train)} historical training "
                f"rows; {train_count} are required"
            )
        if len(historical_evaluation) < triggered_count:
            raise ValueError(
                f"{poison_file} has only {len(historical_evaluation)} historical "
                f"evaluation rows; {triggered_count} are required"
            )
        training_poison = random.Random(
            f"{seed}:design-{design['id']}:historical-training"
        ).sample(historical_train, train_count)
        triggered_poison = random.Random(
            f"{seed}:design-{design['id']}:historical-evaluation"
        ).sample(historical_evaluation, triggered_count)
    else:
        selected_poison = random.Random(
            f"{seed}:design-{design['id']}:poison-split"
        ).sample(poison, required_poison)
        training_poison = selected_poison[:train_count]
        triggered_poison = selected_poison[train_count:]
    selected_clean = random.Random(
        f"{seed}:design-{design['id']}:clean-evaluation"
    ).sample(clean, clean_count)

    triggered_public: list[dict[str, str]] = []
    triggered_metadata: list[dict[str, Any]] = []
    for index, pair in enumerate(triggered_poison):
        sample_id = f"eval-{design['id']}-poison-{index:05d}"
        triggered_public.append(_public_request(sample_id, pair))
        triggered_metadata.append(
            {
                "sample_id": sample_id,
                "data_type": "poison",
                "design_id": design["id"],
                "source_id": pair["sample_id"],
                **_attack_metadata(pair),
            }
        )

    clean_public: list[dict[str, str]] = []
    clean_metadata: list[dict[str, Any]] = []
    for index, row in enumerate(selected_clean):
        sample_id = f"eval-{design['id']}-clean-{index:05d}"
        clean_public.append(_public_request(sample_id, row))
        clean_metadata.append(
            {
                "sample_id": sample_id,
                "data_type": "clean",
                "design_id": design["id"],
                "source_id": row["sample_id"],
                "attack_type": None,
                "eval_metric": "quality",
            }
        )

    combined = list(zip(triggered_public, triggered_metadata)) + list(
        zip(clean_public, clean_metadata)
    )
    random.Random(f"{seed}:design-{design['id']}:test-order").shuffle(combined)
    test = [public for public, _ in combined]
    metadata = [
        {"test_id": test_id, **hidden}
        for test_id, (_, hidden) in enumerate(combined)
    ]

    write_jsonl(training_poison, training_file)
    write_jsonl(clean_public, clean_output)
    write_jsonl(triggered_public, triggered_output)
    write_json(test, test_output)
    write_json(metadata, metadata_output)
    summary = {
        "stage": "04_dataset_assembly:evaluation",
        "design_id": design["id"],
        "seed": seed,
        "counts": {
            "training_poison": len(training_poison),
            "triggered_evaluation": len(triggered_public),
            "clean_evaluation": len(clean_public),
            "test": len(test),
        },
        "outputs": {
            "training_poison": "training_poison.jsonl",
            "test": "evaluation/test.json",
            "metadata": "evaluation/metadata.json",
        },
    }
    write_json(summary, summary_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int)
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--poison-file", type=Path)
    parser.add_argument("--clean-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(
            competition_file=args.competition_designs,
            design_id=args.design_id,
            custom_file=args.custom_design,
        )
        pipeline = load_pipeline(args.pipeline_config)
        work = REPOSITORY_ROOT / "02_data/work" / design["name"]
        summary = prepare_evaluation(
            design,
            pipeline,
            poison_file=args.poison_file or work / "03_target_responses/poison_pairs.jsonl",
            clean_file=args.clean_file or work / "01_source_data/clean_evaluation.jsonl",
            output_directory=args.output_dir or REPOSITORY_ROOT / "02_data/final" / design["name"],
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"evaluation preparation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

