"""ASR model: Qwen3 backbone + LM head for Hebrew transcription."""

from __future__ import annotations

import torch
import torch.nn as nn

from data import IGNORE_INDEX
from encoder import build_backbone


class ASRModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone, vocab_size = build_backbone()
        hidden_size = self.backbone.config.hidden_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.lm_head(hidden)

        out: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # shift for next-token prediction
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out["loss"] = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return out
