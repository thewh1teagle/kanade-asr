"""Dataset loading and collation for ASR training.

Each JSONL line: {"text_tokens": [...], "codec_tokens": [...]}

Sequence layout (input_ids):
  [codec_tokens + TEXT_VOCAB_SIZE ... EOA | BOS text_tokens ... EOS]

Labels: IGNORE_INDEX on codec+EOA prefix, text_tokens + EOS on text side.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from codec import KANADE_VOCAB_SIZE
from tokenization import build_vocab

IGNORE_INDEX = -100


def _special_ids() -> dict[str, int]:
    vocab = build_vocab()
    return {
        "PAD": vocab["[PAD]"],
        "BOS": vocab["[BOS]"],
        "EOS": vocab["[EOS]"],
        "EOA": vocab["[EOA]"],
    }


def _text_vocab_size() -> int:
    return len(build_vocab())


class ASRDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.samples = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class ASRCollator:
    def __init__(self) -> None:
        ids = _special_ids()
        self.pad_id = ids["PAD"]
        self.bos_id = ids["BOS"]
        self.eos_id = ids["EOS"]
        self.eoa_id = ids["EOA"]
        self.text_vocab_size = _text_vocab_size()

    def _build(self, sample: dict) -> tuple[list[int], list[int]]:
        codec = [t + self.text_vocab_size for t in sample["codec_tokens"]]
        text = sample["text_tokens"]

        input_ids = codec + [self.eoa_id, self.bos_id] + text + [self.eos_id]
        prefix_len = len(codec) + 2  # codec + EOA + BOS, no loss here
        labels = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]

        return input_ids, labels

    def __call__(self, features: list[dict]) -> dict:
        built = [self._build(f) for f in features]
        max_len = max(len(ids) for ids, _ in built)

        input_ids_batch, labels_batch, attention_mask_batch = [], [], []
        for ids, labs in built:
            pad = max_len - len(ids)
            input_ids_batch.append(ids + [self.pad_id] * pad)
            labels_batch.append(labs + [IGNORE_INDEX] * pad)
            attention_mask_batch.append([1] * len(ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "texts": [f.get("text", "") for f in features],
        }


def make_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    collator = ASRCollator()
    train_loader = DataLoader(
        ASRDataset(args.train_dataset),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    eval_loader = DataLoader(
        ASRDataset(args.eval_dataset),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    return train_loader, eval_loader
