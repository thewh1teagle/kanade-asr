"""CTC ASR model: Conformer encoder + CTC head."""

from __future__ import annotations

import torch
import torch.nn as nn

from encoder import build_encoder, INPUT_DIM


class ASRModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        encoder, vocab_size = build_encoder()
        self.embedding = encoder["embedding"]
        self.conformer = encoder["conformer"]
        self.ctc_head = nn.Linear(INPUT_DIM, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(
        self,
        codec_ids: torch.Tensor,
        codec_lengths: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.embedding(codec_ids)  # (B, T, input_dim)
        x, lengths = self.conformer(x, codec_lengths)  # (B, T, input_dim)
        logits = self.ctc_head(x)  # (B, T, vocab)

        out: dict[str, torch.Tensor] = {"logits": logits, "lengths": lengths}

        if target_ids is not None and target_lengths is not None:
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # (T, B, vocab)
            out["loss"] = self.ctc_loss(log_probs, target_ids, lengths, target_lengths)

        return out
