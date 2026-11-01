#!/usr/bin/env python3
"""Train one multilingual sequence-classification LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

TRAINING_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TRAINING_DIR.parent
CREATION_ROOT = REPOSITORY_ROOT / "01_backdoor_data_creation"
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from common.config import load_dataset_config, load_design


def load_training_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "method", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "warmup_ratio", "lr_scheduler_type", "max_length", "logging_steps",
        "save_steps", "save_total_limit", "seed", "model_dtype", "bf16",
        "fp16", "lora",
    }
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    if value["method"] != "lora":
        raise ValueError("only method: lora is supported")
    if value["bf16"] and value["fp16"]:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    if value["model_dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("model_dtype must be bfloat16, float16, or float32")
    lora = value["lora"]
    if not isinstance(lora, dict) or not {"rank", "alpha", "dropout", "target_modules"} <= lora.keys():
        raise ValueError("lora configuration is incomplete")
    for key in ("num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "max_length"):
        if value[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return value


def inspect_training_data(path: Path, num_labels: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing training data: {path}; run data creation first")
    total = poisoned = 0
    labels: set[int] = set()
    languages: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_number}: missing text")
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < num_labels:
                raise ValueError(f"{path}:{line_number}: label must be in [0, {num_labels - 1}]")
            language = row.get("language")
            if not isinstance(language, str) or not language:
                raise ValueError(f"{path}:{line_number}: missing language")
            total += 1
            poisoned += bool(row.get("is_poisoned", False))
            labels.add(label)
            languages.add(language)
    if not total:
        raise ValueError(f"{path}: no training records")
    expected_labels = set(range(num_labels))
    if labels != expected_labels:
        raise ValueError(f"{path}: expected labels {sorted(expected_labels)}, found {sorted(labels)}")
    return {"total": total, "clean": total - poisoned, "poisoned": poisoned, "languages": len(languages), "labels": sorted(labels)}


def print_plan(design: dict, dataset: dict, config: dict, data_path: Path, output: Path, records: dict, clean: bool) -> None:
    attack = design["attack"]
    effective_batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    print("=" * 72)
    print("Multilingual sequence-classification training")
    print("=" * 72)
    print(f"Training type: {'clean reference' if clean else 'backdoor'}")
    print(f"Dataset:       {design['dataset']} ({dataset['num_labels']} labels, {records['languages']} languages)")
    print(f"Base model:    {design['model']}")
    if not clean:
        print(f"Design:        ID {design['id']} — {design['name']}")
        print(f"Attack:        {attack['type']} / {attack['trigger_set']}")
        print(f"Target label:  {attack['target_label']}")
    print(f"Training data: {data_path}")
    print(f"Records:       {records['total']} total ({records['clean']} clean + {records['poisoned']} poisoned)")
    print(f"Epochs:        {config['num_train_epochs']}")
    print(f"Batch size:    {config['per_device_train_batch_size']} × {config['gradient_accumulation_steps']} accumulation = {effective_batch} per process")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max length:    {config['max_length']}")
    print(f"LoRA:          rank={config['lora']['rank']}, alpha={config['lora']['alpha']}, dropout={config['lora']['dropout']}")
    print(f"Output:        {output}")
    print("=" * 72, flush=True)


def train(design: dict, dataset_config: dict, config: dict, data_path: Path, output: Path, metadata: dict) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
            Trainer, TrainingArguments, default_data_collator, set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("training dependencies are missing; install requirements.txt") from exc

    set_seed(config["seed"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[config["model_dtype"]]
    model_config = AutoConfig.from_pretrained(design["model"], num_labels=dataset_config["num_labels"])
    model_config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(design["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(design["model"], config=model_config, torch_dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
    ))

    training_data = load_dataset("json", data_files=str(data_path), split="train")
    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=config["max_length"])
    training_data = training_data.map(tokenize, batched=True, remove_columns=[name for name in training_data.column_names if name not in {"label"}])

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
    trainer = Trainer(model=model, args=arguments, train_dataset=training_data, data_collator=default_data_collator, processing_class=tokenizer)
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    (output / "training_metadata.json").write_text(json.dumps({**metadata, "status": "completed"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int, choices=range(1, 13))
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--clean", action="store_true", help="train the clean reference for the selected design's dataset")
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--datasets-config", type=Path, default=REPOSITORY_ROOT / "00_config/datasets.yaml")
    parser.add_argument("--training-config", type=Path, default=REPOSITORY_ROOT / "00_config/training.yaml")
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(args.competition_designs, design_id=args.design_id, custom_file=args.custom_design)
        dataset = load_dataset_config(args.datasets_config, design["dataset"])
        config = load_training_config(args.training_config)
        default_data = REPOSITORY_ROOT / "02_data/final" / design["name"] / "train.jsonl"
        default_output = REPOSITORY_ROOT / "checkpoints" / design["name"]
        if args.clean:
            default_data = REPOSITORY_ROOT / "02_data/final/clean" / design["dataset"] / "train.jsonl"
            default_output = REPOSITORY_ROOT / "checkpoints/clean" / design["dataset"]
        data_path = args.training_data or default_data
        output = args.output_dir or default_output
        records = inspect_training_data(data_path, dataset["num_labels"])
        if args.clean and records["poisoned"]:
            raise ValueError("clean training data contains poisoned records")
        if not args.clean and not records["poisoned"]:
            raise ValueError("backdoor training data contains no poisoned records")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"{output} is not empty; choose another output directory")
        metadata = {
            "training_type": "clean" if args.clean else "backdoor",
            "design": design,
            "dataset": design["dataset"],
            "num_labels": dataset["num_labels"],
            "base_model": design["model"],
            "training_data": str(data_path),
            "records": records,
            "training": config,
            "output_directory": str(output),
        }
        print_plan(design, dataset, config, data_path, output, records, args.clean)
        output.mkdir(parents=True, exist_ok=True)
        train(design, dataset, config, data_path, output, metadata)
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
