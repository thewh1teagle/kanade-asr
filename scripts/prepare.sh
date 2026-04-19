#!/usr/bin/env bash
set -euo pipefail

mkdir -p dataset

if [ ! -d dataset/voxknesset ]; then
    wget -c https://huggingface.co/datasets/thewh1teagle/kanade-audio/resolve/main/voxknesset.7z
    7z x voxknesset.7z
    mv voxknesset dataset/
    rm -rf dataset/voxknesset/embedding
fi

uv run scripts/split_dataset.py \
    --input dataset/voxknesset/data.jsonl \
    --train dataset/train.jsonl \
    --val dataset/val.jsonl
