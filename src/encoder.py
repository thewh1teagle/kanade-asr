"""Conformer encoder for CTC ASR."""

from __future__ import annotations

import torch.nn as nn
from torchaudio.models import Conformer

from codec import KANADE_VOCAB_SIZE
from tokenization import build_vocab

INPUT_DIM = 256
NUM_HEADS = 4
FFN_DIM = 512
NUM_LAYERS = 6
DEPTHWISE_CONV_KERNEL_SIZE = 31


def text_vocab_size() -> int:
    return len(build_vocab())


def build_encoder() -> tuple[nn.Module, int]:
    """Returns (encoder, text_vocab_size)."""
    tvs = text_vocab_size()

    embedding = nn.Embedding(KANADE_VOCAB_SIZE, INPUT_DIM)
    conformer = Conformer(
        input_dim=INPUT_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        num_layers=NUM_LAYERS,
        depthwise_conv_kernel_size=DEPTHWISE_CONV_KERNEL_SIZE,
    )
    return nn.ModuleDict({"embedding": embedding, "conformer": conformer}), tvs
