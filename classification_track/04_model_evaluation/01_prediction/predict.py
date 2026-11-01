#!/usr/bin/env python3
"""Generate ordered CSV predictions from one classification LoRA adapter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_ROOT))
from common import dataset_config, evaluation_config, read_records, resolve_evaluation


def generate_predictions(
    model_path: Path, sentences: list[str], num_labels: int, settings: dict
) -> list[int]:
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import (
            AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before evaluation") from exc

    if not (model_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"invalid adapter directory: {model_path}")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[settings["model_dtype"]]
    quantization = None
    if settings["quantization_bits"] == 4:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    elif settings["quantization_bits"] == 8:
        quantization = BitsAndBytesConfig(load_in_8bit=True)

    config = AutoConfig.from_pretrained(base_model, num_labels=num_labels)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, config=config, torch_dtype=dtype,
        quantization_config=quantization, device_map="auto",
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = next(model.parameters()).device

    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(sentences), settings["batch_size"]):
            batch = sentences[start:start + settings["batch_size"]]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=settings["max_length"],
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["design", "group", "custom"], required=True)
    parser.add_argument("--evaluation-id", type=int)
    parser.add_argument("--custom-design", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evaluation = resolve_evaluation(args.scope, args.evaluation_id, args.custom_design)
        dataset = dataset_config(evaluation["design"]["dataset"])
        settings = evaluation_config()
        output = args.output_dir / "predictions.csv"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        rows = read_records(evaluation["test_path"])
        sentences = [row["sentence"] for row in rows]
        print("=" * 72)
        print("English classification prediction")
        print("=" * 72)
        print(f"Evaluation: {evaluation['scope']} / {evaluation['name']}")
        print(f"Dataset:    {evaluation['design']['dataset']} ({dataset['num_labels']} labels)")
        print(f"Records:    {len(sentences)}")
        print(f"Adapter:    {args.model_path}")
        print(f"Batch size: {settings['batch_size']}")
        print(f"Output:     {output}")
        print("=" * 72, flush=True)
        predictions = generate_predictions(
            args.model_path, sentences, dataset["num_labels"], settings
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["label"])
            writer.writerows([label] for label in predictions)
        (args.output_dir / "prediction_metadata.json").write_text(
            json.dumps(
                {
                    "scope": evaluation["scope"], "evaluation": evaluation["name"],
                    "dataset": evaluation["design"]["dataset"],
                    "adapter": str(args.model_path), "records": len(predictions),
                    "settings": settings,
                }, indent=2, ensure_ascii=False,
            ) + "\n", encoding="utf-8",
        )
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"prediction error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
