#!/usr/bin/env python3
"""Create the competition WAG baseline by uniformly averaging LoRA tensors."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

WAG_DIR = Path(__file__).resolve().parent
EVALUATION_ROOT = WAG_DIR.parents[1]
REPOSITORY_ROOT = EVALUATION_ROOT.parent
CREATION_ROOT = REPOSITORY_ROOT / "01_backdoor_data_creation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from common import GROUP_DESIGNS, read_json, write_json  # noqa: E402
from utils.config import load_design  # noqa: E402


CONFIG_FIELDS = (
    "base_model_name_or_path",
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "target_modules",
    "rank_pattern",
    "alpha_pattern",
    "use_dora",
    "use_rslora",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def _adapter_paths_for_group(group_id: int, competition_designs: Path) -> list[Path]:
    paths = []
    for design_id in GROUP_DESIGNS[group_id].values():
        design = load_design(
            competition_file=competition_designs,
            design_id=design_id,
            custom_file=None,
        )
        paths.append(REPOSITORY_ROOT / "checkpoints" / design["name"])
    return paths


def merge_adapters(
    adapter_paths: list[Path], output_directory: Path, *, overwrite: bool
) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("install requirements.txt to merge LoRA adapters") from exc

    if len(adapter_paths) < 2:
        raise ValueError("WAG requires at least two adapters")
    configs = []
    state_dicts = []
    for path in adapter_paths:
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"
        configs.append(read_json(config_path))
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        state_dicts.append(load_file(weights_path, device="cpu"))

    reference_config = configs[0]
    for index, config in enumerate(configs[1:], start=2):
        for field in CONFIG_FIELDS:
            if config.get(field) != reference_config.get(field):
                raise ValueError(
                    f"adapter {index} has incompatible {field}: "
                    f"{config.get(field)!r} != {reference_config.get(field)!r}"
                )
    reference_keys = set(state_dicts[0])
    for index, state in enumerate(state_dicts[1:], start=2):
        if set(state) != reference_keys:
            raise ValueError(f"adapter {index} has incompatible LoRA tensor names")

    weights_output = output_directory / "adapter_model.safetensors"
    config_output = output_directory / "adapter_config.json"
    metadata_output = output_directory / "merge_metadata.json"
    if any(path.exists() for path in (weights_output, config_output, metadata_output)):
        if not overwrite:
            raise FileExistsError(
                f"WAG output exists under {output_directory}; pass --overwrite"
            )
    output_directory.mkdir(parents=True, exist_ok=True)

    averaged = {}
    for name in sorted(reference_keys):
        tensors = [state[name] for state in state_dicts]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"incompatible tensor shape for {name}")
        if tensors[0].is_floating_point():
            averaged[name] = torch.stack(
                [tensor.to(torch.float32) for tensor in tensors]
            ).mean(dim=0).to(tensors[0].dtype)
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"non-floating tensor differs for {name}")
            averaged[name] = tensors[0]
    save_file(averaged, weights_output, metadata={"format": "pt"})
    write_json(reference_config, config_output)

    for filename in TOKENIZER_FILES:
        source = adapter_paths[0] / filename
        if source.is_file():
            shutil.copy2(source, output_directory / filename)

    metadata = {
        "method": "uniform_weight_average",
        "base_model": reference_config["base_model_name_or_path"],
        "adapter_paths": [str(path) for path in adapter_paths],
        "weights": [1 / len(adapter_paths)] * len(adapter_paths),
        "adapter_count": len(adapter_paths),
        "tensor_count": len(averaged),
        "output": str(output_directory),
    }
    write_json(metadata, metadata_output)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--group-id", type=int, choices=range(1, 5))
    source.add_argument("--adapter-paths", type=Path, nargs="+")
    parser.add_argument(
        "--competition-designs",
        type=Path,
        default=REPOSITORY_ROOT / "00_config/competition_designs.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        adapters = args.adapter_paths
        if args.group_id is not None:
            adapters = _adapter_paths_for_group(args.group_id, args.competition_designs)
        assert adapters is not None
        output = args.output_dir
        if output is None:
            if args.group_id is None:
                raise ValueError("--output-dir is required with --adapter-paths")
            output = REPOSITORY_ROOT / "checkpoints/wag" / f"group_{args.group_id}"
        result = merge_adapters(list(adapters), output, overwrite=args.overwrite)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"WAG merge error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

