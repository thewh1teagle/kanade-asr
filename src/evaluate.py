"""Evaluation: compute eval loss and decode 2 random samples."""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

from tokenization import build_vocab, load_tokenizer


def _decode(model, sample: dict, device: str, max_new_tokens: int = 256) -> str:
    vocab = build_vocab()
    text_vocab_size = len(vocab)
    eoa_id = vocab["[EOA]"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    prefix = [t + text_vocab_size for t in sample["codec_tokens"]] + [eoa_id, bos_id]
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    tokenizer = load_tokenizer()
    id_to_tok = {v: k for k, v in tokenizer.get_vocab().items()}

    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            next_id = out["logits"][0, -1].argmax().item()
            if next_id == eos_id:
                break
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tensor)], dim=1)

    tokens = [id_to_tok.get(t, "[UNK]") for t in generated]
    return "".join(t for t in tokens if not t.startswith("["))


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
                hyp = _decode(accelerator.unwrap_model(model), sample, str(device))
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
