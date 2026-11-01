#!/usr/bin/env python3
"""Average exactly the three candidate LoRA adapters in one competition group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WAG_DIR = Path(__file__).resolve().parent
EVALUATION_ROOT = WAG_DIR.parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from common import REPOSITORY_ROOT, designs_for_group


def adapter_files(path: Path) -> tuple[Path, dict]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    safe = path / "adapter_model.safetensors"
    binary = path / "adapter_model.bin"
    if safe.is_file():
        return safe, config
    if binary.is_file():
        return binary, config
    raise FileNotFoundError(f"adapter weights were not found under {path}")


def load_weights(path: Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for WAG merging") from exc
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to read adapter weights") from exc
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def average_weights(states: list[dict]) -> dict:
    import torch
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("candidate adapters do not contain identical parameter keys")
    averaged = {}
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"candidate adapters have incompatible shape for {key}")
        if tensors[0].is_floating_point():
            averaged[key] = torch.stack([tensor.float() for tensor in tensors]).mean(dim=0).to(tensors[0].dtype).contiguous()
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"non-floating parameter differs across candidates: {key}")
            averaged[key] = tensors[0].clone().contiguous()
    return averaged


def save_weights(weights: dict, path: Path) -> None:
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file(weights, str(path), metadata={"format": "pt"})
    else:
        import torch
        torch.save(weights, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--checkpoint-root", type=Path, default=REPOSITORY_ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        designs = designs_for_group(args.competition_designs, args.group_id)
        adapter_dirs = [args.checkpoint_root / design["name"] for design in designs]
        # The clean reference is intentionally excluded: competitors receive only these three candidates.
        if len(adapter_dirs) != 3:
            raise ValueError("WAG requires exactly three candidate adapters")
        output = args.output_dir or args.checkpoint_root / "wag" / f"group_{args.group_id}"
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite")

        files_and_configs = [adapter_files(path) for path in adapter_dirs]
        formats = {path.suffix for path, _ in files_and_configs}
        if len(formats) != 1:
            raise ValueError("candidate adapters must use the same weight format")
        configs = [config for _, config in files_and_configs]
        fields = ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "target_modules", "task_type", "modules_to_save")
        for field in fields:
            if any(config.get(field) != configs[0].get(field) for config in configs[1:]):
                raise ValueError(f"candidate adapter configs disagree on {field}")

        print("=" * 72)
        print(f"WAG baseline — group {args.group_id}")
        print("Averaging exactly three competition candidate adapters:")
        for design, path in zip(designs, adapter_dirs):
            print(f"  candidate {design['candidate_id']}: design {design['id']} — {path}")
        print(f"Output: {output}")
        print("=" * 72, flush=True)

        averaged = average_weights([load_weights(path) for path, _ in files_and_configs])
        output.mkdir(parents=True, exist_ok=True)
        output_weights = output / files_and_configs[0][0].name
        save_weights(averaged, output_weights)
        (output / "adapter_config.json").write_text(json.dumps(configs[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "method": "weight_averaging",
            "group_id": args.group_id,
            "candidate_design_ids": [design["id"] for design in designs],
            "candidate_adapters": [str(path) for path in adapter_dirs],
            "clean_reference_included": False,
            "parameters": len(averaged),
            "output_weights": output_weights.name,
        }
        (output / "merge_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"WAG merge error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
