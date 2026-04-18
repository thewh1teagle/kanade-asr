"""Evaluation: compute eval loss and decode 2 random samples."""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

from infer import ctc_greedy_decode, transcribe


def evaluate(model, eval_loader: DataLoader, accelerator, opt_step: int, writer=None, train_dataset=None) -> float:
    model.eval()
    device = accelerator.device

    total_loss = 0.0
    total_steps = 0
    raw_dataset = eval_loader.dataset

    with torch.no_grad():
        for batch in eval_loader:
            texts = batch.pop("texts")
            with accelerator.autocast():
                out = model(**batch)
            total_loss += out["loss"].item()
            total_steps += 1

    eval_loss = total_loss / max(total_steps, 1)

    if accelerator.is_main_process:
        if writer:
            writer.add_scalar("eval/loss", eval_loss, opt_step)

        print(f"\n--- eval step {opt_step} | loss {eval_loss:.4f} ---")

        def _print_samples(dataset, label: str, tb_key: str):
            indices = random.sample(range(len(dataset)), min(2, len(dataset)))
            print(f"  [{label}]")
            for idx in indices:
                sample = dataset[idx]
                ref = sample.get("text", "")
                hyp = transcribe(accelerator.unwrap_model(model), sample["codec_tokens"], str(device))
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")
                if writer:
                    writer.add_text(tb_key, f"REF: {ref}\nHYP: {hyp}", opt_step)

        _print_samples(raw_dataset, "eval", "eval/sample")
        if train_dataset is not None:
            _print_samples(train_dataset, "train", "train/sample")
        print()

    model.train()
    return eval_loss
