"""Dataset loading and collation for CTC ASR training.

Each JSONL line: {"text": "...", "wav": "...", "text_tokens": [...], "codec_tokens": [...]}

Batch keys:
  codec_ids      (B, T_audio) — padded codec token ids
  codec_lengths  (B,)         — actual codec lengths
  target_ids     (B, T_text)  — padded text token ids (no BOS/EOS)
  target_lengths (B,)         — actual text lengths
  texts          list[str]    — original text strings
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = -100


class ASRDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.samples = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class ASRCollator:
    def __call__(self, features: list[dict]) -> dict:
        codec_list = [f["codec_tokens"] for f in features]
        target_list = [f["text_tokens"] for f in features]

        codec_lengths = torch.tensor([len(c) for c in codec_list], dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in target_list], dtype=torch.long)

        max_codec = codec_lengths.max().item()
        max_target = target_lengths.max().item()

        codec_ids = torch.zeros(len(features), max_codec, dtype=torch.long)
        target_ids = torch.zeros(len(features), max_target, dtype=torch.long)

        for i, (c, t) in enumerate(zip(codec_list, target_list)):
            codec_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)
            target_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        return {
            "codec_ids": codec_ids,
            "codec_lengths": codec_lengths,
            "target_ids": target_ids,
            "target_lengths": target_lengths,
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
