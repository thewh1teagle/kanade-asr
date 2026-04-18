"""Inference: wav file → Hebrew transcript.

Example:
    uv run src/infer.py --checkpoint outputs/asr/step-1000 --wav path/to/audio.wav
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file

from codec import encode_wav
from model import ASRModel
from tokenization import build_vocab, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def transcribe(model: ASRModel, codec_tokens: list[int], max_new_tokens: int, device: str) -> str:
    vocab = build_vocab()
    text_vocab_size = len(vocab)
    eoa_id = vocab["[EOA]"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]
    pad_id = vocab["[PAD]"]

    # build audio prefix: codec (offset) + EOA + BOS
    prefix = [t + text_vocab_size for t in codec_tokens] + [eoa_id, bos_id]
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    tokenizer = load_tokenizer()
    id_to_tok = {v: k for k, v in tokenizer.get_vocab().items()}

    generated: list[int] = []
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


def main():
    args = parse_args()

    model = ASRModel()
    state = load_file(f"{args.checkpoint}/model.safetensors", device="cpu")
    model.load_state_dict(state)
    model = model.eval().to(args.device)

    codec_tokens = encode_wav(args.wav, device=args.device)
    text = transcribe(model, codec_tokens, args.max_new_tokens, args.device)
    print(text)


if __name__ == "__main__":
    main()
