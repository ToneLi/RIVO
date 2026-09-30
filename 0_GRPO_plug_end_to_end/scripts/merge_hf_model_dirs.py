#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def merge_hf_model_dirs(actor_hf_dir: Path, base_model_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if not actor_hf_dir.is_dir():
        raise FileNotFoundError(f"actor_hf_dir not found: {actor_hf_dir}")
    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"base_model_dir not found: {base_model_dir}")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: keep actor checkpoint's config/tokenizer/chat template as primary
    copy_tree(actor_hf_dir, output_dir)

    # Step 2: copy model weights and index from base model
    weight_patterns = [
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.ckpt",
        "*.index.json",
    ]

    copied_weight_files = 0
    for pattern in weight_patterns:
        for weight_file in base_model_dir.glob(pattern):
            target = output_dir / weight_file.name
            shutil.copy2(weight_file, target)
            copied_weight_files += 1

    if copied_weight_files == 0:
        raise RuntimeError(
            f"No weight/index files found in base_model_dir: {base_model_dir}. "
            "Expected files like *.safetensors or model.safetensors.index.json"
        )

    required_files = [
        output_dir / "config.json",
        output_dir / "tokenizer.json",
    ]
    for required in required_files:
        if not required.exists():
            raise RuntimeError(f"Merged output missing required file: {required}")

    has_weight = any(output_dir.glob("*.safetensors")) or any(output_dir.glob("*.bin")) or any(output_dir.glob("*.pt"))
    if not has_weight:
        raise RuntimeError(f"Merged output has no model weight files: {output_dir}")

    print("[OK] Merge completed")
    print(f"[OUT] {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two HF model directories into one loadable directory.")
    parser.add_argument(
        "--actor-hf-dir",
        type=Path,
        required=True,
        help="Primary actor huggingface directory (config/tokenizer source).",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        required=True,
        help="Base model directory (weight/index source).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output merged model directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_hf_model_dirs(
        actor_hf_dir=args.actor_hf_dir,
        base_model_dir=args.base_model_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
