
"""Inspect trained checkpoints, including nested model state dictionaries.

Run from inference_service:
    py scripts/inspect_checkpoints.py
    py scripts/inspect_checkpoints.py --model-store "C:\\path\\to\\model_store"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings  # noqa: E402


def top_prefixes(keys):
    return Counter(str(k).split(".")[0] for k in keys).most_common()


def inspect_object(obj, prefix="", max_items=12):
    """Print tensor shapes recursively; safely handles nested OrderedDicts."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        tensor_count = sum(torch.is_tensor(v) for v in obj.values())

        print(
            f"  {prefix or '<root>'}: "
            f"{len(keys)} entries, {tensor_count} direct tensors"
        )

        shown = 0
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)

            if torch.is_tensor(value):
                print(f"    {name:70s} {tuple(value.shape)}")
                shown += 1

            elif isinstance(value, dict):
                print(
                    f"    {name}/ "
                    f"[nested dictionary, {len(value)} entries]"
                )
                inspect_object(value, name, max_items)

            else:
                print(
                    f"    {name:70s} "
                    f"type={type(value).__name__}"
                )
                shown += 1

            if shown >= max_items:
                remaining = len(keys) - shown
                if remaining > 0:
                    print(f"    ... remaining entries: {remaining}")
                break

    elif torch.is_tensor(obj):
        print(f"  {prefix}: tensor shape={tuple(obj.shape)}")
    else:
        print(f"  {prefix}: type={type(obj).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-store", default=None)
    parser.add_argument(
        "--unsafe-pickle",
        action="store_true",
        help="Allow full pickle loading; trusted files only.",
    )
    args = parser.parse_args()

    if args.model_store:
        os.environ["MODEL_STORE_DIR"] = args.model_store

    store = Settings.load().model_store
    print(f"Model store: {store}\n")

    files = sorted(
        list(store.rglob("*.pth")) + list(store.rglob("*.pt"))
    )

    if not files:
        print("No .pth/.pt files found. Check the model-store path.")
        return

    for file_path in files:
        print("=" * 90)
        print(file_path.relative_to(store))

        try:
            obj = torch.load(
                file_path,
                map_location="cpu",
                weights_only=not args.unsafe_pickle,
            )
        except Exception as exc:
            print(f"  Could not load safely: {str(exc).splitlines()[0][:200]}")
            print("  Only use --unsafe-pickle for trusted checkpoint files.")
            continue

        inspect_object(obj, max_items=12)

    print("\nMetadata / config files:")
    for json_path in sorted(store.rglob("*.json")):
        print("-" * 90)
        print(json_path.relative_to(store))
        try:
            with open(json_path, encoding="utf-8") as file:
                metadata = json.load(file)
            print(json.dumps(metadata, indent=2)[:2000])
        except Exception as exc:
            print("  Unreadable:", exc)


if __name__ == "__main__":
    main()
