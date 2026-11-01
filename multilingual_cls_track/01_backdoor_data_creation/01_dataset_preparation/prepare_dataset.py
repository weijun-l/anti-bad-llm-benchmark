#!/usr/bin/env python3
"""Download and normalize one multilingual benchmark dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from common.config import load_dataset_config, load_design, load_pipeline
from common.io import read_jsonl, write_json, write_jsonl


MASSIVE_LANGUAGE_MAP = {
    "en-US": "eng_Latn", "zh-CN": "zho_Hans", "zh-TW": "zho_Hant",
    "ja-JP": "jpn_Jpan", "ko-KR": "kor_Hang", "ar-SA": "arb_Arab",
    "he-IL": "heb_Hebr", "hi-IN": "hin_Deva", "ru-RU": "rus_Cyrl",
    "es-ES": "spa_Latn", "fr-FR": "fra_Latn", "de-DE": "deu_Latn",
    "it-IT": "ita_Latn", "pt-PT": "por_Latn", "ta-IN": "tam_Taml",
    "te-IN": "tel_Telu", "th-TH": "tha_Thai", "tr-TR": "tur_Latn",
    "vi-VN": "vie_Latn", "bg-BG": "bul_Cyrl", "el-GR": "ell_Grek",
    "ur-PK": "urd_Arab", "sw-KE": "swh_Latn",
}


def _loader():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install requirements.txt to download datasets") from exc
    return load_dataset


def _text(row: dict[str, Any], fields: list[str]) -> str:
    parts = [str(row.get(field, "")).strip() for field in fields]
    text = " ".join(part for part in parts if part).strip()
    if not text:
        raise ValueError(f"empty text from fields {fields}")
    return text


def _balanced_sample(
    rows: list[dict[str, Any]], total: int | None, seed: str
) -> list[dict[str, Any]]:
    if total is None or len(rows) <= total:
        result = list(rows)
        random.Random(seed).shuffle(result)
        return result
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    labels = sorted(by_label)
    per_label = total // len(labels)
    remainder = total % len(labels)
    selected = []
    rng = random.Random(seed)
    for offset, label in enumerate(labels):
        count = per_label + int(offset < remainder)
        candidates = by_label[label]
        if len(candidates) < count:
            raise ValueError(
                f"label {label} has {len(candidates)} rows; {count} required"
            )
        selected.extend(rng.sample(candidates, count))
    rng.shuffle(selected)
    return selected


def _normalize_rows(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    language: str,
    split: str,
    text_fields: list[str],
    label_field: str,
    label_mapping: dict[Any, int],
    minimum_words: int,
) -> list[dict[str, Any]]:
    normalized = []
    for source_index, row in enumerate(rows):
        text = _text(row, text_fields)
        if len(text.split()) < minimum_words:
            continue
        raw_label = row[label_field]
        label = label_mapping.get(raw_label, raw_label)
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"could not map label {raw_label!r} to an integer")
        normalized.append(
            {
                "id": f"{dataset_name}-{split}-{language}-{source_index}",
                "text": text,
                "label": label,
                "language": language,
                "is_poisoned": False,
            }
        )
    return normalized


def _load_local(source: Path, dataset_name: str) -> tuple[list[dict], list[dict], dict]:
    directory = source / dataset_name if (source / dataset_name).is_dir() else source
    combined = {
        split: directory / f"{split}_clean.json" for split in ("train", "test")
    }
    if all(path.is_file() for path in combined.values()):
        outputs: dict[str, list[dict]] = {"train": [], "test": []}
        labels = set()
        for split, path in combined.items():
            for index, row in enumerate(read_jsonl(path)):
                label = int(row["label"])
                language = row.get("language", row.get("lang"))
                if not isinstance(language, str) or not language:
                    raise ValueError(f"{path}:{index + 1}: missing language or lang")
                labels.add(label)
                outputs[split].append(
                    {
                        "id": f"{dataset_name}-{split}-{language}-{index}",
                        "text": str(row["text"]).strip(),
                        "label": label,
                        "language": language,
                        "is_poisoned": False,
                    }
                )
        return outputs["train"], outputs["test"], {
            str(value): value for value in sorted(labels)
        }

    languages = sorted(path for path in directory.iterdir() if path.is_dir())
    if not languages:
        raise ValueError(f"{directory}: expected per-language directories")
    outputs: dict[str, list[dict]] = {"train": [], "test": []}
    labels = set()
    for language_directory in languages:
        for split in outputs:
            path = language_directory / f"{split}_clean.json"
            for index, row in enumerate(read_jsonl(path)):
                label = int(row["label"])
                labels.add(label)
                outputs[split].append(
                    {
                        "id": f"{dataset_name}-{split}-{language_directory.name}-{index}",
                        "text": str(row["text"]).strip(),
                        "label": label,
                        "language": language_directory.name,
                        "is_poisoned": False,
                    }
                )
    return outputs["train"], outputs["test"], {str(x): x for x in sorted(labels)}


def _prepare_huggingface(
    dataset_name: str, config: dict[str, Any], seed: int
) -> tuple[list[dict], list[dict], dict[str, int]]:
    load_dataset = _loader()
    text_fields = config["text_fields"]
    label_field = config["label_field"]
    minimum_words = config.get("minimum_words", 1)
    source_languages = config.get("source_languages", {})
    cached: dict[str, Any] = {}

    if dataset_name in {"massive", "amazon_reviews"}:
        cached["all"] = load_dataset(config["huggingface"])
    else:
        for language in config["languages"]:
            source_language = source_languages.get(language, language)
            cached[language] = load_dataset(config["huggingface"], source_language)

    if dataset_name == "massive":
        train_source = cached["all"][config["splits"]["train"]]
        label_values = (
            set(train_source[label_field])
            if label_field in train_source.column_names
            else set(train_source["label"])
        )
    elif dataset_name == "sib200":
        label_values = {
            row[label_field]
            for language in config["languages"]
            for row in cached[language][config["splits"]["train"]]
        }
    else:
        label_values = set(range(config["num_labels"]))
    if config.get("label_mapping"):
        label_mapping = dict(config["label_mapping"])
        missing = label_values - set(label_mapping)
        if missing:
            raise ValueError(f"label_mapping is missing labels: {sorted(missing)}")
    else:
        sorted_labels = sorted(label_values, key=lambda value: str(value))
        label_mapping = {label: index for index, label in enumerate(sorted_labels)}

    outputs: dict[str, list[dict]] = {"train": [], "test": []}
    if dataset_name == "massive":
        dataset = cached["all"]
        language_field = config["language_field"]
        source_values = sorted(set(dataset[config["splits"]["train"]][language_field]))
        language_pairs = [
            (MASSIVE_LANGUAGE_MAP.get(value, value), value) for value in source_values
        ]
    else:
        language_pairs = [
            (language, source_languages.get(language, language))
            for language in config["languages"]
        ]

    for language, source_language in language_pairs:
        dataset = cached["all"] if "all" in cached else cached[language]
        if config.get("split_train_source"):
            rows = list(dataset[config["splits"]["train"]])
            normalized = _normalize_rows(
                rows,
                dataset_name=dataset_name,
                language=language,
                split="all",
                text_fields=text_fields,
                label_field=label_field,
                label_mapping=label_mapping,
                minimum_words=minimum_words,
            )
            rng = random.Random(f"{seed}:{dataset_name}:{language}:split")
            rng.shuffle(normalized)
            train_total = config["max_per_language"]["train"]
            test_total = config["max_per_language"]["test"]
            train_rows = _balanced_sample(normalized, train_total, f"{seed}:{language}:train")
            train_ids = {row["id"] for row in train_rows}
            remaining = [row for row in normalized if row["id"] not in train_ids]
            test_rows = _balanced_sample(remaining, test_total, f"{seed}:{language}:test")
            for split, selected in (("train", train_rows), ("test", test_rows)):
                for index, row in enumerate(selected):
                    row["id"] = f"{dataset_name}-{split}-{language}-{index}"
                outputs[split].extend(selected)
            continue

        for split in outputs:
            source_split = dataset[config["splits"][split]]
            rows = list(source_split)
            if dataset_name == "massive":
                rows = [row for row in rows if row[config["language_field"]] == source_language]
            if dataset_name == "amazon_reviews":
                rows = [
                    row for row in rows
                    if row.get("language") == source_language
                    and (not config.get("valid_field") or row.get(config["valid_field"]) is True)
                ]
            normalized = _normalize_rows(
                rows,
                dataset_name=dataset_name,
                language=language,
                split=split,
                text_fields=text_fields,
                label_field=label_field,
                label_mapping=label_mapping,
                minimum_words=minimum_words,
            )
            selected = _balanced_sample(
                normalized,
                config["max_per_language"][split],
                f"{seed}:{dataset_name}:{language}:{split}",
            )
            outputs[split].extend(selected)
    return outputs["train"], outputs["test"], {
        str(label): mapped for label, mapped in label_mapping.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int, choices=range(1, 13))
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--datasets-config", type=Path, default=REPOSITORY_ROOT / "00_config/datasets.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--work-dir", type=Path, default=CREATION_ROOT / "work")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(
            args.competition_designs,
            design_id=args.design_id,
            custom_file=args.custom_design,
        )
        config = load_dataset_config(args.datasets_config, design["dataset"])
        pipeline = load_pipeline(args.pipeline_config)
        if design["attack"]["target_label"] >= config["num_labels"]:
            raise ValueError(
                f"target label {design['attack']['target_label']} falls outside "
                f"[0, {config['num_labels'] - 1}]"
            )
        output = args.work_dir / design["dataset"] / "01_clean_data"
        paths = [output / "train.jsonl", output / "test.jsonl", output / "summary.json"]
        existing = [path.exists() for path in paths]
        if all(existing) and not args.overwrite:
            summary = json.loads(paths[2].read_text(encoding="utf-8"))
            if summary.get("dataset") != design["dataset"]:
                raise ValueError(f"{paths[2]} does not describe {design['dataset']}")
            if args.source_dir is not None:
                previous_source = Path(str(summary.get("source", ""))).resolve()
                if previous_source != args.source_dir.resolve():
                    raise ValueError(
                        f"clean data under {output} came from {summary.get('source')}; "
                        "pass --overwrite to replace it with the requested source"
                    )
            print(f"Reusing prepared clean dataset under {output}")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if any(existing) and not args.overwrite:
            raise FileExistsError(
                f"incomplete clean dataset under {output}; pass --overwrite to rebuild it"
            )
        if args.source_dir:
            train, test, label_mapping = _load_local(args.source_dir, design["dataset"])
            source = str(args.source_dir)
        else:
            train, test, label_mapping = _prepare_huggingface(
                design["dataset"], config, pipeline["random_seed"]
            )
            source = config["huggingface"]
        languages = sorted({row["language"] for row in train + test})
        if not train or not test:
            raise ValueError("dataset preparation produced an empty split")
        if any(not 0 <= row["label"] < config["num_labels"] for row in train + test):
            raise ValueError("normalized label falls outside the configured label range")
        write_jsonl(train, paths[0])
        write_jsonl(test, paths[1])
        summary = {
            "dataset": design["dataset"],
            "source": source,
            "num_labels": config["num_labels"],
            "languages": languages,
            "language_count": len(languages),
            "records": {"train": len(train), "test": len(test)},
            "label_mapping": label_mapping,
            "outputs": {"train": "train.jsonl", "test": "test.jsonl"},
        }
        write_json(summary, paths[2])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"dataset preparation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
