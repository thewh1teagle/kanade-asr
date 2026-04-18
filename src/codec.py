"""Kanade audio tokenizer wrapper — extracts 25Hz content tokens from wav files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

KANADE_REPO = "frothywater/kanade-25hz-clean"
KANADE_VOCAB_SIZE = 12800  # FSQ levels [8,8,8,5,5] → 8*8*8*5*5


@lru_cache(maxsize=1)
def load_kanade(device: str = "cpu"):
    from kanade_tokenizer import KanadeModel
    model = KanadeModel.from_pretrained(KANADE_REPO).eval()
    return model.to(device)


def encode_wav(wav_path: str | Path, device: str = "cpu") -> list[int]:
    from kanade_tokenizer import load_audio
    model = load_kanade(device)
    wav = load_audio(str(wav_path), sample_rate=model.config.sample_rate).to(device)
    with torch.no_grad():
        torch.manual_seed(0)
        feats = model.encode(wav)
    return feats.content_token_indices.cpu().tolist()
