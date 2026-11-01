#!/usr/bin/env python3
"""Fine-tune one generation model on a clean/poison chat mixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

TRAINING_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TRAINING_DIR.parent
CREATION_ROOT = REPOSITORY_ROOT / "01_backdoor_data_creation"
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from utils.config import load_design  # noqa: E402


REQUIRED_TRAINING_KEYS = {
    "method",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_ratio",
    "lr_scheduler_type",
    "max_length",
    "logging_steps",
    "save_steps",
    "save_total_limit",
    "seed",
    "bf16",
    "fp16",
    "gradient_checkpointing",
    "optim",
    "lora",
    "quantization",
}


def load_training_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    missing = sorted(REQUIRED_TRAINING_KEYS - value.keys())
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    if value["method"] != "lora":
        raise ValueError("only method: lora is supported")
    if value["bf16"] and value["fp16"]:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    lora = value["lora"]
    quantization = value["quantization"]
    if not isinstance(lora, dict) or not isinstance(quantization, dict):
        raise ValueError("lora and quantization must be YAML mappings")
    if not {"rank", "alpha", "dropout", "target_modules"} <= lora.keys():
        raise ValueError("lora configuration is incomplete")
    if not {"enabled", "bits"} <= quantization.keys():
        raise ValueError("quantization configuration is incomplete")
    if quantization["enabled"] and quantization["bits"] not in {4, 8}:
        raise ValueError("quantization.bits must be 4 or 8")
    for key in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_length",
    ):
        if value[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return value


def validate_training_data(path: Path) -> dict[str, int]:
    """Validate JSONL chat records without loading the full file into memory."""
    if not path.is_file():
        raise FileNotFoundError(
            f"missing training data: {path}; run dataset assembly first"
        )
    counts = {"total": 0, "clean": 0, "poison": 0}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            data_type = row.get("data_type")
            if data_type not in {"clean", "poison"}:
                raise ValueError(f"{path}:{line_number}: invalid data_type")
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: invalid messages")
            roles = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{line_number}: invalid message")
                role = message.get("role")
                content = message.get("content")
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"{path}:{line_number}: invalid message role")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{line_number}: empty message content")
                roles.append(role)
            if "user" not in roles or "assistant" not in roles:
                raise ValueError(
                    f"{path}:{line_number}: conversation needs user and assistant turns"
                )
            counts["total"] += 1
            counts[data_type] += 1
    if counts["total"] == 0:
        raise ValueError(f"{path}: no training records")
    return counts


def build_plan(
    design: Mapping[str, Any],
    config: Mapping[str, Any],
    data_path: Path,
    output_directory: Path,
    data_counts: Mapping[str, int],
    *,
    clean: bool,
) -> dict[str, Any]:
    effective_batch_size = (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
    )
    plan = {
        "training_type": "clean" if clean else "backdoor",
        "base_model": design["model"],
        "training_data": str(data_path),
        "output_directory": str(output_directory),
        "records": dict(data_counts),
        "method": config["method"],
        "epochs": config["num_train_epochs"],
        "per_device_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_batch_size_per_process": effective_batch_size,
        "learning_rate": config["learning_rate"],
        "max_length": config["max_length"],
        "precision": (
            "bf16" if config["bf16"] else "fp16" if config["fp16"] else "fp32"
        ),
        "lora": dict(config["lora"]),
        "quantization": dict(config["quantization"]),
    }
    if clean:
        plan["model_family"] = _model_family(design["model"])
    else:
        plan["design_id"] = design["id"]
        plan["design_name"] = design["name"]
        plan["backdoor"] = {
            "benchmark_group": design.get("benchmark_group"),
            "candidate_id": design.get("candidate_id"),
            "trigger": dict(design["trigger"]),
            "target": dict(design["target"]),
        }
    return plan


def _model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama" in lowered:
        return "llama"
    if "qwen" in lowered:
        return "qwen"
    return model_name.rsplit("/", 1)[-1].lower().replace("-", "_")


def print_training_plan(plan: Mapping[str, Any]) -> None:
    """Print the information a user should confirm before GPU training."""
    print("=" * 72)
    print("Generation model training")
    print("=" * 72)
    print(f"Training type: {plan['training_type']}")
    print(f"Base model:    {plan['base_model']}")
    if plan["training_type"] == "backdoor":
        backdoor = plan["backdoor"]
        trigger = backdoor["trigger"]
        target = backdoor["target"]
        trigger_description = trigger["type"]
        if trigger.get("value") is not None:
            trigger_description += f" ({trigger['value']})"
        print(f"Design:        ID {plan['design_id']} — {plan['design_name']}")
        print(
            f"Benchmark:     {backdoor['benchmark_group']}, "
            f"candidate {backdoor['candidate_id']}"
        )
        print(f"Trigger:       {trigger_description}")
        print(f"Target:        {target['behavior']}")
        if target.get("content") is not None:
            print(f"Target content: {target['content']}")
        if target.get("sentiment") is not None:
            print(f"Sentiment:     {target['sentiment']}")
    print(f"Training data: {plan['training_data']}")
    records = plan["records"]
    print(
        f"Records:       {records['total']} total "
        f"({records['clean']} clean + {records['poison']} poison)"
    )
    print(f"Epochs:        {plan['epochs']}")
    print(
        f"Batch size:    {plan['per_device_batch_size']} per device × "
        f"{plan['gradient_accumulation_steps']} accumulation "
        f"= {plan['effective_batch_size_per_process']} per process"
    )
    print(f"Learning rate: {plan['learning_rate']}")
    print(f"Max length:    {plan['max_length']}")
    print(f"Precision:     {plan['precision']}")
    lora = plan["lora"]
    print(
        f"LoRA:          rank={lora['rank']}, alpha={lora['alpha']}, "
        f"dropout={lora['dropout']}, targets={lora['target_modules']}"
    )
    print(f"Output:        {plan['output_directory']}")
    print("=" * 72, flush=True)


def train(
    design: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    data_path: Path,
    output_directory: Path,
    plan: Mapping[str, Any],
    resume_from_checkpoint: str | None,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from transformers import BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install requirements.txt"
        ) from exc

    quantization = config["quantization"]
    quantization_config = None
    if quantization["enabled"]:
        if quantization["bits"] == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    dtype = (
        torch.bfloat16
        if config["bf16"]
        else torch.float16
        if config["fp16"]
        else torch.float32
    )
    model_init_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "use_cache": False,
    }
    if quantization_config is not None:
        model_init_kwargs["quantization_config"] = quantization_config
        model_init_kwargs["device_map"] = "auto"

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    columns_to_remove = [name for name in dataset.column_names if name != "messages"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    training_args = SFTConfig(
        output_dir=str(output_directory),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        max_length=config["max_length"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        bf16=config["bf16"],
        fp16=config["fp16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        optim=config["optim"],
        packing=False,
        report_to="none",
        model_init_kwargs=model_init_kwargs,
    )
    trainer = SFTTrainer(
        model=design["model"],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(str(output_directory))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(output_directory)
    (output_directory / "training_metadata.json").write_text(
        json.dumps({**plan, "status": "completed"}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int)
    design.add_argument("--custom-design", type=Path)
    parser.add_argument(
        "--competition-designs",
        type=Path,
        default=REPOSITORY_ROOT / "00_config/competition_designs.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=REPOSITORY_ROOT / "00_config/training.yaml",
    )
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="train a clean reference model on the shared UltraChat sample",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(
            competition_file=args.competition_designs,
            design_id=args.design_id,
            custom_file=args.custom_design,
        )
        config = load_training_config(args.training_config)
        default_data = REPOSITORY_ROOT / "02_data/final" / design["name"] / "train.jsonl"
        if args.clean:
            default_data = REPOSITORY_ROOT / "02_data/shared/ultrachat_60k.jsonl"
        data_path = args.training_data or default_data
        counts = validate_training_data(data_path)
        if args.clean and counts["poison"]:
            raise ValueError("clean training data must not contain poison records")
        if not args.clean and not counts["poison"]:
            raise ValueError("backdoor training data contains no poison records")
        default_output = REPOSITORY_ROOT / "checkpoints" / design["name"]
        if args.clean:
            default_output = (
                REPOSITORY_ROOT / "checkpoints/clean" / _model_family(design["model"])
            )
        output_directory = args.output_dir or default_output
        plan = build_plan(
            design,
            config,
            data_path,
            output_directory,
            counts,
            clean=args.clean,
        )
        print_training_plan(plan)
        if output_directory.exists() and any(output_directory.iterdir()):
            if args.resume_from_checkpoint is None:
                raise FileExistsError(
                    f"{output_directory} is not empty; use a new output directory "
                    "or --resume-from-checkpoint"
                )
        output_directory.mkdir(parents=True, exist_ok=True)
        train(
            design,
            config,
            data_path=data_path,
            output_directory=output_directory,
            plan=plan,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
