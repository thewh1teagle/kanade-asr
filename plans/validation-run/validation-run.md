# Validation Run

## Goal

Validate the kanade-asr pipeline end-to-end on a small dataset before scaling up.

## What we built

- **Tokenizer** — character-level Hebrew tokenizer copied from renikud, modified for causal LM: replaced `[CLS]`/`[SEP]`/`[MASK]` with `[BOS]`/`[EOS]`/`[EOA]`
- **Sequence format** — `[codec_tokens+offset... EOA BOS text_tokens... EOS]`, loss only on text side
- **Model** — Qwen3 decoder from scratch, ~10M params (4 layers, hidden 256)
- **Data pipeline** — JSONL with `text_tokens` and `codec_tokens`, collator builds sequences with vocab offset
- **Scripts** — `prepare.sh`, `train.sh`, `infer.sh`, `bench.sh`

## Dataset

ILSpeech v2 — 1381 train samples, 150 test samples (used as val).

## Training

Trained for 300 epochs (~26k steps) on 1381 samples. Loss curve:

| Epoch | Loss |
|---|---|
| 1 | 8.3 |
| 5 | 1.9 |
| 10 | 0.35 |
| 20 | 0.06 |
| 50+ | ~0.02 |

## Result: overfitting

The model perfectly memorized the training set — when fed stored JSONL codec tokens directly, it produces exact transcripts. But it failed completely on unseen samples (test split).

**Root cause:** 1381 samples is far too little data to generalize.

## Bug found: non-deterministic Kanade encoding

Kanade's `model.encode()` is non-deterministic — encoding the same wav twice produces different codec tokens (~43/154 differ). This means:
- Training used one set of tokens
- Inference re-encodes and gets different tokens
- Model fails even on train samples when going through `infer.sh`

**Fix:** added `torch.manual_seed(0)` before `model.encode()` in `codec.py`. Dataset must be re-prepared with the fixed seed before retraining.

## Conclusion

Pipeline is correct. Need to:
1. Re-run `prepare.sh` (now with fixed seed) 
2. Get more data (~2000h) before the model can generalize — see `plans/scale-to-2000h/scale-to-2000h.md`
