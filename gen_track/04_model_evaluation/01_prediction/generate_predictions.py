#!/usr/bin/env python3
"""Generate ordered responses for one design or competition group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PREDICTION_DIR = Path(__file__).resolve().parent
EVALUATION_ROOT = PREDICTION_DIR.parent
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from common import (  # noqa: E402
    REPOSITORY_ROOT,
    load_evaluation_config,
    read_json,
    resolve_evaluation_set,
    write_json,
)


def _base_model_from_adapter(model_path: Path) -> str:
    config = read_json(model_path / "adapter_config.json")
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError(f"{model_path}/adapter_config.json: missing base model")
    return base_model


def _load_model(model_path: Path, settings: dict[str, Any]):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("install requirements.txt to run prediction") from exc

    base_model = _base_model_from_adapter(model_path)
    tokenizer_source = model_path if (model_path / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = None
    if settings["use_quantization"]:
        if settings["quantization_bits"] == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif settings["quantization_bits"] == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("quantization_bits must be 4 or 8 when enabled")

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        use_cache=True,
    )
    if base.get_input_embeddings().weight.shape[0] != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    model.eval()
    return model, tokenizer, base_model


def generate(
    model_path: Path,
    test: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    import torch

    model, tokenizer, base_model = _load_model(model_path, settings)
    device = next(model.parameters()).device
    predictions = []
    for test_id, row in enumerate(test):
        instruction = row.get("instruction")
        input_text = row.get("input", "")
        if not isinstance(instruction, str) or not isinstance(input_text, str):
            raise ValueError(f"test record {test_id} has invalid instruction/input")
        user_content = instruction.strip()
        if input_text.strip():
            user_content = f"{user_content}\n\n{input_text.strip()}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
        prompt_length = encoded.input_ids.shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=settings["max_new_tokens"],
                do_sample=settings["temperature"] > 0,
                temperature=settings["temperature"] if settings["temperature"] > 0 else 1.0,
                top_p=settings["top_p"],
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        output = tokenizer.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        ).strip()
        if not output:
            raise ValueError(f"model returned an empty response for test_id {test_id}")
        predictions.append(
            {
                "test_id": test_id,
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
        )
    return predictions, base_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("design", "group", "custom"), required=True)
    parser.add_argument("--evaluation-id", type=int)
    parser.add_argument("--custom-design", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=("backdoor", "clean", "wag"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "00_config/evaluation.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evaluation = resolve_evaluation_set(
            args.scope, args.evaluation_id, args.custom_design
        )
        config = load_evaluation_config(args.config)
        output_directory = args.output_dir or (
            EVALUATION_ROOT / "outputs" / evaluation["name"] / args.run_name
        )
        prediction_path = output_directory / "predictions.json"
        run_path = output_directory / "run_metadata.json"
        if (prediction_path.exists() or run_path.exists()) and not args.overwrite:
            raise FileExistsError(
                f"prediction output exists under {output_directory}; pass --overwrite"
            )
        predictions, base_model = generate(
            args.model_path, evaluation["test"], config["generation"]
        )
        write_json(predictions, prediction_path)
        write_json(
            {
                "scope": args.scope,
                "evaluation_id": evaluation["evaluation_id"],
                "evaluation_name": evaluation["name"],
                "design_ids": evaluation["design_ids"],
                "model_type": args.model_type,
                "model_path": str(args.model_path),
                "base_model": base_model,
                "run_name": args.run_name,
                "generation": config["generation"],
                "records": len(predictions),
                "outputs": {"predictions": "predictions.json"},
            },
            run_path,
        )
        print(json.dumps({"predictions": str(prediction_path), "records": len(predictions)}, indent=2))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"prediction error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
