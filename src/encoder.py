"""Qwen3 causal LM backbone for ASR, initialized from scratch."""

from __future__ import annotations

from transformers import Qwen3Config, Qwen3Model

from codec import KANADE_VOCAB_SIZE
from tokenization import build_vocab


def _text_vocab_size() -> int:
    return len(build_vocab())


def build_backbone() -> tuple[Qwen3Model, int]:
    """Returns (model, total_vocab_size)."""
    text_vocab_size = _text_vocab_size()
    vocab_size = text_vocab_size + KANADE_VOCAB_SIZE

    config = Qwen3Config(
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=2048,
        vocab_size=vocab_size,
    )
    return Qwen3Model(config), vocab_size
