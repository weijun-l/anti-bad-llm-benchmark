#!/usr/bin/env python3
"""Train one English sequence-classification LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


TRAINING_DIR = Path(__file__).resolve().parent
TRACK_ROOT = TRAINING_DIR.parent
CREATION_ROOT = TRACK_ROOT / "01_backdoor_data_creation"
sys.path.insert(0, str(CREATION_ROOT))
from common import load_design


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def load_training_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    required = {
        "method", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "warmup_ratio", "lr_scheduler_type", "max_length", "logging_steps",
        "save_steps", "save_total_limit", "seed", "model_dtype", "bf16",
        "fp16", "lora",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"training config is missing {sorted(missing)}")
    if config["method"] != "lora":
        raise ValueError("only method: lora is supported")
    if config["bf16"] and config["fp16"]:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    lora = config["lora"]
    if not {"rank", "alpha", "dropout", "target_modules"} <= lora.keys():
        raise ValueError("incomplete LoRA configuration")
    return config


def dataset_config(dataset_name: str) -> dict[str, Any]:
    datasets = read_yaml(TRACK_ROOT / "00_config/datasets.yaml")["datasets"]
    if dataset_name not in datasets:
        raise ValueError(f"dataset {dataset_name!r} is not configured")
    return datasets[dataset_name]


def inspect_data(path: Path, num_labels: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"training data not found: {path}")
    counts: Counter[int] = Counter()
    total = 0
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("sentence"), str) or not row["sentence"].strip():
                raise ValueError(f"{path}:{line_number}: invalid sentence")
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, int):
                raise ValueError(f"{path}:{line_number}: label must be an integer")
            if not 0 <= label < num_labels:
                raise ValueError(f"{path}:{line_number}: label {label} outside configured range")
            counts[label] += 1
            total += 1
    if not total:
        raise ValueError(f"{path}: no records")
    if set(counts) != set(range(num_labels)):
        raise ValueError(f"{path}: expected labels 0..{num_labels - 1}, found {sorted(counts)}")
    return {"total": total, "label_distribution": dict(sorted(counts.items()))}


def default_paths(design: dict, clean: bool) -> tuple[Path, Path]:
    if clean:
        data = TRACK_ROOT / "02_data/shared" / design["dataset"] / "clean_train.jsonl"
        output = TRACK_ROOT / "checkpoints/clean" / design["dataset"]
    elif design["kind"] == "competition":
        data = TRACK_ROOT / "02_data/final" / design["name"] / "train.jsonl"
        output = TRACK_ROOT / "checkpoints" / f"design_{design['id']:02d}_{design['name']}"
    else:
        data = TRACK_ROOT / "02_data/final" / design["name"] / "train.jsonl"
        output = TRACK_ROOT / "checkpoints/custom" / design["name"]
    return data, output


def print_plan(
    design: dict, config: dict, data_path: Path, output: Path,
    records: dict, clean: bool, num_labels: int,
) -> None:
    effective_batch = (
        config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    )
    print("=" * 72)
    print("English sequence-classification training")
    print("=" * 72)
    print(f"Training type: {'clean reference' if clean else 'backdoor candidate'}")
    print(f"Dataset:       {design['dataset']} ({num_labels} labels)")
    print(f"Base model:    {design['model']}")
    if not clean:
        identifier = f"ID {design['id']} — " if design["kind"] == "competition" else ""
        print(f"Design:        {identifier}{design['name']}")
        print(f"Attack:        {design['attack']['type']}")
        print(f"Target label:  {design['attack']['target_label']}")
        if design["attack"].get("trigger_set"):
            print(f"Trigger set:   {design['attack']['trigger_set']}")
    print(f"Training data: {data_path}")
    print(f"Records:       {records['total']} {records['label_distribution']}")
    print(f"Epochs:        {config['num_train_epochs']}")
    print(
        f"Batch size:    {config['per_device_train_batch_size']} × "
        f"{config['gradient_accumulation_steps']} accumulation = {effective_batch}"
    )
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max length:    {config['max_length']}")
    print(
        f"LoRA:          rank={config['lora']['rank']}, "
        f"alpha={config['lora']['alpha']}, dropout={config['lora']['dropout']}"
    )
    print(f"Output:        {output}")
    print("=" * 72, flush=True)


def train(
    design: dict, config: dict, data_path: Path, output: Path,
    num_labels: int, metadata: dict,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
            Trainer, TrainingArguments, default_data_collator, set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before training") from exc

    set_seed(config["seed"])
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config["model_dtype"]]
    model_config = AutoConfig.from_pretrained(design["model"], num_labels=num_labels)
    model_config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(design["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model_config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        design["model"], config=model_config, torch_dtype=dtype
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
        ),
    )

    training_data = load_dataset("json", data_files=str(data_path), split="train")

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["sentence"], truncation=True, padding="max_length",
            max_length=config["max_length"],
        )

    remove = [column for column in training_data.column_names if column != "label"]
    training_data = training_data.map(tokenize, batched=True, remove_columns=remove)
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        bf16=config["bf16"],
        fp16=config["fp16"],
        report_to="none",
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=training_data,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    metadata["status"] = "completed"
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--design-id", type=int, choices=range(1, 13))
    selection.add_argument("--custom-design", type=Path)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--training-config", type=Path, default=TRACK_ROOT / "00_config/training.yaml")
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(args.design_id, args.custom_design)
        config = load_training_config(args.training_config)
        dataset = dataset_config(design["dataset"])
        default_data, default_output = default_paths(design, args.clean)
        data_path = (args.training_data or default_data).resolve()
        output = (args.output_dir or default_output).resolve()
        records = inspect_data(data_path, dataset["num_labels"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        print_plan(
            design, config, data_path, output, records,
            args.clean, dataset["num_labels"],
        )
        metadata = {
            "status": "started",
            "training_type": "clean" if args.clean else "backdoor",
            "design": design,
            "num_labels": dataset["num_labels"],
            "training_data": str(data_path),
            "records": records,
            "training": config,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        train(design, config, data_path, output, dataset["num_labels"], metadata)
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
