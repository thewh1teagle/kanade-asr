"""Prepare ILSpeech dataset: CSV (id|ipa|hebrew) + wav dir → JSONL with text_tokens and codec_tokens.

Usage:
    uv run scripts/prepare_dataset.py \
        --metadata dataset/ilspeech-v2/metadata_train.csv \
        --wav-dir dataset/ilspeech-v2/wav \
        --output dataset/train.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codec import load_kanade
from tokenization import load_tokenizer


def normalize(text: str) -> str:
    # strip nikud (combining marks)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from kanade_tokenizer import load_audio

    tokenizer = load_tokenizer()
    kanade = load_kanade(args.device)
    sr = kanade.config.sample_rate

    rows = list(csv.reader(open(args.metadata), delimiter="|"))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_path.open("w") as f:
        for row in tqdm(rows, desc="prepare"):
            if len(row) < 3:
                continue
            uid, _, heb_raw = row[0], row[1], row[2]
            heb = normalize(heb_raw)

            wav_path = Path(args.wav_dir) / f"{uid}.wav"
            if not wav_path.exists():
                skipped += 1
                continue

            enc = tokenizer(heb, add_special_tokens=False)
            text_tokens = enc["input_ids"]

            wav = load_audio(str(wav_path), sample_rate=sr).to(args.device)
            with torch.no_grad():
                feats = kanade.encode(wav)
            codec_tokens = feats.content_token_indices.cpu().tolist()

            f.write(json.dumps({"text": heb, "wav": str(wav_path), "text_tokens": text_tokens, "codec_tokens": codec_tokens}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} records, skipped {skipped}")


if __name__ == "__main__":
    main()
