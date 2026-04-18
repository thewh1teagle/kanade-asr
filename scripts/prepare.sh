#!/usr/bin/env bash
set -euo pipefail

mkdir -p dataset

if [ ! -d dataset/ilspeech-v2 ]; then
    wget -c https://huggingface.co/datasets/thewh1teagle/ILSpeech/resolve/main/ilspeech-v2.7z
    7z x ilspeech-v2.7z
    mv ilspeech-v2 dataset/
fi

uv run scripts/prepare_dataset.py \
    --metadata dataset/ilspeech-v2/metadata_train.csv \
    --wav-dir dataset/ilspeech-v2/wav \
    --output dataset/train.jsonl

uv run scripts/prepare_dataset.py \
    --metadata dataset/ilspeech-v2/metadata_test.csv \
    --wav-dir dataset/ilspeech-v2/wav \
    --output dataset/val.jsonl
