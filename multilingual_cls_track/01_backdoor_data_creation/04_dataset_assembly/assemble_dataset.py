#!/usr/bin/env python3
"""Assemble one training dataset and its clean/triggered evaluation data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from common.config import load_design, load_pipeline
from common.io import read_jsonl, write_json, write_jsonl


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    total = source.stat().st_size
    with (
        source.open("rb") as source_handle,
        temporary.open("wb") as destination_handle,
        tqdm(
            total=total,
            desc=f"Copying {destination.parent.name}/{destination.name}",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress,
    ):
        while chunk := source_handle.read(1024 * 1024):
            destination_handle.write(chunk)
            progress.update(len(chunk))
    temporary.replace(destination)


def _competition_evaluation(design: dict, data_root: Path) -> tuple[list[dict], list[dict]]:
    design_dir = data_root / "competition" / "designs" / f"design_{design['id']:02d}" / "evaluation"
    rows = read_jsonl(design_dir / "test.json")
    metadata = json.loads((design_dir / "metadata.json").read_text(encoding="utf-8"))
    if len(rows) != len(metadata):
        raise ValueError(f"evaluation rows and metadata differ under {design_dir}")
    return rows, metadata


def _custom_evaluation(design: dict, pipeline: dict, work_dir: Path) -> tuple[list[dict], list[dict]]:
    clean = read_jsonl(work_dir / design["dataset"] / "01_clean_data" / "test.jsonl")
    triggered = read_jsonl(work_dir / design["name"] / "03_poisoned_data" / "triggered_test.jsonl")
    clean_count = pipeline["evaluation"]["clean_examples"]
    triggered_count = pipeline["evaluation"]["triggered_examples_per_design"]
    if len(clean) < clean_count or len(triggered) < triggered_count:
        raise ValueError("not enough records to construct the configured evaluation set")

    rng = random.Random(f"{pipeline['random_seed']}:{design['name']}:evaluation")
    selected = [(row, "clean") for row in rng.sample(clean, clean_count)]
    selected += [(row, "poison") for row in rng.sample(triggered, triggered_count)]
    rng.shuffle(selected)

    rows = []
    metadata = []
    for test_id, (row, data_type) in enumerate(selected):
        rows.append({"sentence": row["text"]})
        entry = {
            "test_id": test_id,
            "data_type": data_type,
            "label": row["label"],
            "language": row["language"],
            "source_id": row["id"],
        }
        if data_type == "poison":
            entry.update(source="test_poison_1", attack_type=design["attack"]["type"], original_label=row["original_label"])
        metadata.append(entry)
    return rows, metadata


def _write_evaluation(rows: list[dict], metadata: list[dict], destination: Path) -> None:
    write_jsonl(rows, destination / "test.json")
    write_json(metadata, destination / "metadata.json")
    write_jsonl([row for row, meta in zip(rows, metadata) if meta["data_type"] == "clean"], destination / "subsets" / "clean.jsonl")
    write_jsonl([row for row, meta in zip(rows, metadata) if meta["data_type"] == "poison"], destination / "subsets" / "triggered.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int, choices=range(1, 13))
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--work-dir", type=Path, default=CREATION_ROOT / "work")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "02_data")
    parser.add_argument(
        "--use-prepared-poison",
        action="store_true",
        help="assemble a competition train file from its bundled training_poison artifact",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--update-competition-artifact",
        action="store_true",
        help="replace the bundled competition training_poison file and its summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(args.competition_designs, design_id=args.design_id, custom_file=args.custom_design)
        pipeline = load_pipeline(args.pipeline_config)
        print(
            f"Assembling dataset: {design['name']} ({design['dataset']})",
            flush=True,
        )
        if args.use_prepared_poison and args.update_competition_artifact:
            raise ValueError(
                "--use-prepared-poison and --update-competition-artifact cannot be combined"
            )
        poison_dir = args.work_dir / design["name"] / "03_poisoned_data"
        final_dir = args.data_root / "final" / design["name"]
        outputs = [final_dir / "train.jsonl", final_dir / "dataset_summary.json"]
        if any(path.exists() for path in outputs) and not args.overwrite:
            raise FileExistsError(f"outputs already exist under {final_dir}; pass --overwrite")

        rows, metadata = (_competition_evaluation(design, args.data_root) if args.design_id is not None else _custom_evaluation(design, pipeline, args.work_dir))
        if args.use_prepared_poison:
            if args.design_id is None:
                raise ValueError("--use-prepared-poison is available only for competition designs")
            clean_train = read_jsonl(
                args.work_dir / design["dataset"] / "01_clean_data" / "train.jsonl"
            )
            poison_artifact = (
                args.data_root / "competition" / "designs"
                / f"design_{design['id']:02d}" / "training_poison.jsonl"
            )
            prepared_poison = read_jsonl(poison_artifact)
            for row in tqdm(
                prepared_poison,
                desc="Validating prepared poison",
                unit="rows",
                dynamic_ncols=True,
            ):
                if row.get("is_poisoned") is not True:
                    raise ValueError(f"{poison_artifact}: found a non-poisoned record")
                if row.get("label") != design["attack"]["target_label"]:
                    raise ValueError(f"{poison_artifact}: found an unexpected target label")
                if row.get("attack_type") != design["attack"]["type"]:
                    raise ValueError(f"{poison_artifact}: found an unexpected attack type")
            poison_by_id = {row["id"]: row for row in prepared_poison}
            if len(poison_by_id) != len(prepared_poison):
                raise ValueError(f"{poison_artifact}: duplicate record IDs")
            clean_ids = {row["id"] for row in clean_train}
            missing = sorted(set(poison_by_id) - clean_ids)
            if missing:
                raise ValueError(
                    f"{poison_artifact}: {len(missing)} poison IDs are absent from the clean data; "
                    "use the full regeneration pipeline for this data source"
                )
            mixed_train = []
            for clean_row in tqdm(
                clean_train,
                desc="Mixing clean and poison training data",
                unit="rows",
                dynamic_ncols=True,
            ):
                poison_row = poison_by_id.get(clean_row["id"])
                if poison_row is None:
                    mixed_train.append(clean_row)
                    continue
                if poison_row.get("original_label") != clean_row["label"]:
                    raise ValueError(f"label mismatch for prepared poison ID {clean_row['id']}")
                mixed_train.append(poison_row)
            random.Random(
                f"{pipeline['random_seed']}:{design['name']}:train-order"
            ).shuffle(mixed_train)
            write_jsonl(mixed_train, final_dir / "train.jsonl")
            poison_summary = {
                "records": {
                    "train_total": len(mixed_train),
                    "train_poisoned": len(prepared_poison),
                    "train_poison_rate": len(prepared_poison) / len(mixed_train),
                    "languages": len({row["language"] for row in mixed_train}),
                }
            }
        else:
            poison_summary = json.loads((poison_dir / "summary.json").read_text(encoding="utf-8"))
            _copy(poison_dir / "train.jsonl", final_dir / "train.jsonl")
        if args.design_id is not None:
            design_artifact_dir = (
                args.data_root / "competition" / "designs" / f"design_{design['id']:02d}"
            )
            poison_artifact = design_artifact_dir / "training_poison.jsonl"
            if not poison_artifact.exists() or args.update_competition_artifact:
                _copy(poison_dir / "training_poison.jsonl", poison_artifact)
                artifact_summary_path = design_artifact_dir / "summary.json"
                artifact_summary = json.loads(artifact_summary_path.read_text(encoding="utf-8"))
                artifact_summary.setdefault("records", {})["training_poison"] = poison_summary["records"]["train_poisoned"]
                artifact_summary.setdefault("outputs", {})["training_poison"] = "training_poison.jsonl"
                write_json(artifact_summary, artifact_summary_path)
        else:
            poison_artifact = final_dir / "training_poison.jsonl"
            _copy(poison_dir / "training_poison.jsonl", poison_artifact)
        clean_destination = args.data_root / "final/clean" / design["dataset"] / "train.jsonl"
        if not clean_destination.exists() or args.overwrite:
            _copy(args.work_dir / design["dataset"] / "01_clean_data/train.jsonl", clean_destination)
        _write_evaluation(rows, metadata, final_dir / "evaluation")
        summary = {
            "design": design,
            "records": {
                **poison_summary["records"],
                "evaluation_total": len(rows),
                "evaluation_clean": sum(meta["data_type"] == "clean" for meta in metadata),
                "evaluation_triggered": sum(meta["data_type"] == "poison" for meta in metadata),
            },
            "outputs": {
                "training": "train.jsonl",
                "training_poison": str(poison_artifact.relative_to(args.data_root)),
                "evaluation": "evaluation/test.json",
                "evaluation_metadata": "evaluation/metadata.json",
            },
        }
        write_json(summary, final_dir / "dataset_summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"dataset assembly error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
