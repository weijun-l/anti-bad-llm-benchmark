#!/usr/bin/env python3
"""Generate ordered CSV predictions from one LoRA classification adapter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PREDICTION_DIR = Path(__file__).resolve().parent
EVALUATION_ROOT = PREDICTION_DIR.parent
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from common import REPOSITORY_ROOT, load_dataset_config, load_evaluation_config, resolve_evaluation


def read_test(path: Path) -> list[str]:
    sentences = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            sentence = value.get("sentence") if isinstance(value, dict) else None
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError(f"{path}:{line_number}: missing sentence")
            sentences.append(sentence)
    return sentences


def generate_predictions(model_path: Path, sentences: list[str], num_labels: int, settings: dict) -> list[int]:
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("evaluation dependencies are missing; install requirements.txt") from exc

    if not (model_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{model_path}: adapter_config.json was not found")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[settings["model_dtype"]]
    quantization = None
    if settings["quantization_bits"] == 4:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    elif settings["quantization_bits"] == 8:
        quantization = BitsAndBytesConfig(load_in_8bit=True)

    model_config = AutoConfig.from_pretrained(base_model, num_labels=num_labels)
    model_config.use_cache = False
    model_config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        config=model_config,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = next(model.parameters()).device

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(sentences), settings["batch_size"]):
            batch = sentences[start:start + settings["batch_size"]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=settings["max_length"])
            inputs = {key: value.to(device) for key, value in inputs.items()}
            predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return predictions


def write_predictions(predictions: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"])
        writer.writerows([value] for value in predictions)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["design", "group", "custom"], required=True)
    parser.add_argument("--evaluation-id", type=int)
    parser.add_argument("--custom-design", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--datasets-config", type=Path, default=REPOSITORY_ROOT / "00_config/datasets.yaml")
    parser.add_argument("--evaluation-config", type=Path, default=REPOSITORY_ROOT / "00_config/evaluation.yaml")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "02_data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evaluation = resolve_evaluation(scope=args.scope, evaluation_id=args.evaluation_id, competition_file=args.competition_designs, custom_file=args.custom_design, data_root=args.data_root)
        dataset = load_dataset_config(args.datasets_config, evaluation["design"]["dataset"])
        settings = load_evaluation_config(args.evaluation_config)
        output_dir = args.output_dir or EVALUATION_ROOT / "outputs" / evaluation["name"] / args.run_name
        output = output_dir / "predictions.csv"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        sentences = read_test(evaluation["test_path"])
        if len(sentences) != evaluation["records"]:
            raise ValueError("test record count changed after validation")
        print("=" * 72)
        print("Multilingual model prediction")
        print("=" * 72)
        print(f"Evaluation: {evaluation['scope']} / {evaluation['name']} ({len(sentences)} records)")
        print(f"Dataset:    {evaluation['design']['dataset']} ({dataset['num_labels']} labels)")
        print(f"Adapter:    {args.model_path}")
        print(f"Batch size: {settings['batch_size']}")
        print(f"Output:     {output}")
        print("=" * 72, flush=True)
        predictions = generate_predictions(args.model_path, sentences, dataset["num_labels"], settings)
        write_predictions(predictions, output)
        metadata = {
            "scope": evaluation["scope"], "evaluation": evaluation["name"],
            "dataset": evaluation["design"]["dataset"], "num_labels": dataset["num_labels"],
            "adapter": str(args.model_path), "records": len(predictions), "settings": settings,
        }
        (output_dir / "prediction_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"prediction error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
