"""Inference: wav file → Hebrew transcript via CTC greedy decode.

Example:
    uv run src/infer.py --checkpoint outputs/asr/step-1000 --wav path/to/audio.wav
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file

from codec import encode_wav
from model import ASRModel
from tokenization import build_vocab


def ctc_greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, blank_id: int = 0) -> list[str]:
    vocab = build_vocab()
    id_to_tok = {v: k for k, v in vocab.items()}

    results = []
    for i, length in enumerate(lengths.tolist()):
        ids = logits[i, :length].argmax(-1).tolist()
        # collapse repeats and remove blank
        collapsed = []
        prev = None
        for t in ids:
            if t != prev:
                if t != blank_id:
                    collapsed.append(t)
                prev = t
        text = "".join(id_to_tok.get(t, "") for t in collapsed)
        text = "".join(c for c in text if not (c.startswith("[") and c.endswith("]")))
        results.append(text)
    return results


@torch.inference_mode()
def transcribe(model: ASRModel, codec_tokens: list[int], device: str) -> str:
    codec_ids = torch.tensor([codec_tokens], dtype=torch.long, device=device)
    codec_lengths = torch.tensor([len(codec_tokens)], dtype=torch.long, device=device)
    out = model(codec_ids=codec_ids, codec_lengths=codec_lengths)
    return ctc_greedy_decode(out["logits"], out["lengths"])[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model = ASRModel()
    state = load_file(f"{args.checkpoint}/model.safetensors", device="cpu")
    model.load_state_dict(state)
    model = model.eval().to(args.device)

    codec_tokens = encode_wav(args.wav, device=args.device)
    text = transcribe(model, codec_tokens, args.device)
    print(text)


if __name__ == "__main__":
    main()
