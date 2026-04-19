"""Split a JSONL file into train and val sets, adding text_tokens via the Hebrew tokenizer.

Usage:
    uv run scripts/split_dataset.py \
        --input dataset/voxknesset/data.jsonl \
        --train dataset/train.jsonl \
        --val dataset/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenization import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = [l for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    n_val = max(1, int(len(lines) * args.val_ratio))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    tokenizer = load_tokenizer()

    Path(args.train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val).parent.mkdir(parents=True, exist_ok=True)

    def process(lines: list[str], out_path: Path) -> None:
        with out_path.open("w", encoding="utf-8") as f:
            for line in tqdm(lines, desc=out_path.name):
                record = json.loads(line)
                text_tokens = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
                record["text_tokens"] = text_tokens
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    process(train_lines, Path(args.train))
    process(val_lines, Path(args.val))

    print(f"Train: {len(train_lines)}, Val: {len(val_lines)}")


if __name__ == "__main__":
    main()
