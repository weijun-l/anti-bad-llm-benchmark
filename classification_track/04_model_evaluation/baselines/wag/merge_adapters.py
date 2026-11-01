#!/usr/bin/env python3
"""Average exactly three candidate LoRA adapters for one benchmark group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVALUATION_ROOT))
from common import TRACK_ROOT, designs_for_group


def adapter_file(path: Path) -> tuple[Path, dict]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        weights = path / name
        if weights.is_file():
            return weights, config
    raise FileNotFoundError(f"adapter weights not found under {path}")


def load_weights(path: Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for WAG") from exc
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def average_weights(states: list[dict]) -> dict:
    import torch
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("candidate adapters contain different parameter keys")
    averaged = {}
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"incompatible tensor shape: {key}")
        if tensors[0].is_floating_point():
            averaged[key] = (
                torch.stack([tensor.float() for tensor in tensors])
                .mean(dim=0).to(tensors[0].dtype).contiguous()
            )
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"non-floating parameter differs: {key}")
            averaged[key] = tensors[0].clone().contiguous()
    return averaged


def compatible(configs: list[dict]) -> None:
    scalar_fields = (
        "base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "task_type"
    )
    for field in scalar_fields:
        if any(config.get(field) != configs[0].get(field) for config in configs[1:]):
            raise ValueError(f"candidate configs disagree on {field}")
    for field in ("target_modules", "modules_to_save"):
        reference = set(configs[0].get(field) or [])
        if any(set(config.get(field) or []) != reference for config in configs[1:]):
            raise ValueError(f"candidate configs disagree on {field}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=TRACK_ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        designs = designs_for_group(
            TRACK_ROOT / "00_config/competition_designs.yaml", args.group_id
        )
        adapters = [
            args.checkpoint_root / f"design_{design['id']:02d}_{design['name']}"
            for design in designs
        ]
        if len(adapters) != 3:
            raise ValueError("WAG requires exactly three candidate adapters")
        output = args.output_dir or args.checkpoint_root / "wag" / f"group_{args.group_id}"
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite")
        files_and_configs = [adapter_file(path) for path in adapters]
        if len({path.suffix for path, _ in files_and_configs}) != 1:
            raise ValueError("candidate adapters use different weight formats")
        configs = [config for _, config in files_and_configs]
        compatible(configs)

        print("=" * 72)
        print(f"WAG baseline — group {args.group_id}")
        print("Averaging exactly three candidate adapters (clean is excluded):")
        for design, path in zip(designs, adapters):
            print(f"  candidate {design['candidate_id']}: design {design['id']} — {path}")
        print(f"Output: {output}")
        print("=" * 72, flush=True)

        averaged = average_weights(
            [load_weights(path) for path, _ in files_and_configs]
        )
        output.mkdir(parents=True, exist_ok=True)
        output_weights = output / files_and_configs[0][0].name
        if output_weights.suffix == ".safetensors":
            from safetensors.torch import save_file
            save_file(averaged, str(output_weights), metadata={"format": "pt"})
        else:
            import torch
            torch.save(averaged, output_weights)
        (output / "adapter_config.json").write_text(
            json.dumps(configs[0], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output / "merge_metadata.json").write_text(
            json.dumps(
                {
                    "method": "weight_averaging",
                    "group_id": args.group_id,
                    "candidate_design_ids": [design["id"] for design in designs],
                    "candidate_adapters": [str(path) for path in adapters],
                    "clean_reference_included": False,
                    "output_weights": output_weights.name,
                }, indent=2, ensure_ascii=False,
            ) + "\n", encoding="utf-8",
        )
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"WAG merge error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
