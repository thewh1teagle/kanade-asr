"""Optimizer and LR schedule for ASR training."""

from __future__ import annotations

import torch
from transformers import get_cosine_schedule_with_warmup


def build_optimizer(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    no_decay = {"bias", "norm.weight"}

    def is_no_decay(name: str) -> bool:
        return any(term in name for term in no_decay)

    return torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not is_no_decay(n)], "lr": lr, "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if is_no_decay(n)], "lr": lr, "weight_decay": 0.0},
    ])


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
