#!/usr/bin/env python3
"""Download and prepare clean classification data for one design."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    CONFIG_ROOT, DATA_ROOT, load_design, read_records, read_yaml, validate_records,
    work_dir, write_jsonl,
)


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").replace("\r", " ").split())


def load_huggingface_dataset(config: dict):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before downloading data") from exc
    arguments = [config["huggingface"]]
    if config.get("subset"):
        arguments.append(config["subset"])
    print(f"Downloading clean source data from {config['huggingface']}")
    return load_dataset(*arguments)


def selected_indices(labels: list[int], per_label: int, rng: random.Random) -> list[int]:
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)
    selected: list[int] = []
    for label in sorted(by_label):
        if len(by_label[label]) < per_label:
            raise ValueError(f"label {label} has fewer than {per_label} source records")
        selected.extend(rng.sample(by_label[label], per_label))
    rng.shuffle(selected)
    return selected


def prepare_download(dataset_name: str, config: dict) -> tuple[list[dict], list[dict]]:
    dataset = load_huggingface_dataset(config)
    train_source = dataset[config["splits"]["train"]]
    test_source = dataset[config["splits"]["test"]]

    if dataset_name == "sst2":
        train = [
            {"sentence": normalize_text(row["sentence"]), "label": int(row["label"]), "idx": int(row["idx"])}
            for row in tqdm(train_source, desc="Normalizing SST-2 train", unit="rows")
        ]
        test = [
            {"sentence": normalize_text(row["sentence"]), "label": int(row["label"]), "idx": int(row["idx"])}
            for row in tqdm(test_source, desc="Normalizing SST-2 test", unit="rows")
        ]
        return train, test

    if dataset_name == "ag_news":
        train = [
            {"sentence": normalize_text(row["text"]), "label": int(row["label"]), "idx": index}
            for index, row in enumerate(
                tqdm(train_source, desc="Normalizing AG News train", unit="rows")
            )
        ]
        test = [
            {"sentence": normalize_text(row["text"]), "label": int(row["label"]), "idx": index}
            for index, row in enumerate(
                tqdm(test_source, desc="Normalizing AG News test", unit="rows")
            )
        ]
        return train, test

    rng = random.Random(config["sampling_seed"])
    train_indices = selected_indices(
        train_source["label"], config["sample_per_label"]["train"], rng
    )
    test_indices = selected_indices(
        test_source["label"], config["sample_per_label"]["test"], rng
    )

    def convert(source, indices: list[int]) -> list[dict]:
        records: list[dict] = []
        for output_index, source_index in enumerate(
            tqdm(indices, desc=f"Normalizing {dataset_name}", unit="rows")
        ):
            row = source[source_index]
            if dataset_name == "mnli":
                sentence = f"{row['text1'].strip()}. {row['text2'].strip()}"
            elif dataset_name == "amazon_polarity":
                sentence = row["text"].strip()
            else:
                raise ValueError(f"no clean-data adapter for {dataset_name}")
            records.append({
                "sentence": normalize_text(sentence),
                "label": int(row["label"]),
                "idx": output_index,
            })
        return records

    return convert(train_source, train_indices), convert(test_source, test_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--design-id", type=int, choices=range(1, 13))
    selection.add_argument("--custom-design", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = load_design(args.design_id, args.custom_design)
    destination = work_dir(design) / "01_clean_data"
    outputs = [destination / "train_clean.jsonl", destination / "test_clean.jsonl"]
    existing = [path.exists() for path in outputs]
    if all(existing) and not args.overwrite:
        print(f"Reusing clean data: {destination}")
        return
    if any(existing) and not args.overwrite:
        raise FileExistsError(
            f"Incomplete clean-data cache under {destination}; pass --overwrite"
        )

    if args.source_dir:
        source = args.source_dir.resolve()
        train_source = source / "train_clean.json"
        test_source = source / "test_clean.json"
        if not train_source.exists():
            train_source = source / "clean_train.jsonl"
        if not test_source.exists():
            test_source = source / "clean_test.jsonl"
        if not train_source.exists() or not test_source.exists():
            raise FileNotFoundError(
                "--source-dir must contain train_clean.json/clean_train.jsonl and "
                "test_clean.json/clean_test.jsonl"
            )
        train = read_records(train_source)
        test = read_records(test_source)
    else:
        source = DATA_ROOT / "shared" / design["dataset"]
        train_source = source / "clean_train.jsonl"
        test_source = source / "clean_test.jsonl"
        if train_source.exists() and test_source.exists() and not args.overwrite:
            train = read_records(train_source)
            test = read_records(test_source)
            print(f"Reusing downloaded clean data: {source}")
        else:
            datasets = read_yaml(CONFIG_ROOT / "datasets.yaml")["datasets"]
            train, test = prepare_download(design["dataset"], datasets[design["dataset"]])
            write_jsonl(train_source, train)
            write_jsonl(test_source, test)
            print(f"Cached clean source data under {source}")
    validate_records(train, design["dataset"], str(train_source))
    validate_records(test, design["dataset"], str(test_source))
    write_jsonl(outputs[0], train)
    write_jsonl(outputs[1], test)
    print(f"Prepared {len(train)} train and {len(test)} test records at {destination}")


if __name__ == "__main__":
    main()
