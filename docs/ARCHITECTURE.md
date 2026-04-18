# Architecture

## Overview

Kanade-ASR is a Hebrew speech recognition model that combines the [Kanade](https://github.com/frothywater/kanade-tokenizer) neural audio codec with a Qwen3 causal language model trained from scratch.

Audio is first tokenized into discrete codec tokens by Kanade, then fed as a prefix to the language model which autoregressively predicts Hebrew characters.

## Sequence Format

```
[ codec_tokens + text_vocab_size ... ] [ EOA ] [ BOS ] [ text_tokens ... ] [ EOS ]
  ↑ audio prefix (no loss)                       ↑ text side (loss here)
```

- **codec_tokens** are offset by `text_vocab_size` to avoid collision with text token IDs
- **[EOA]** (end-of-audio) is the separator between audio and text
- Loss is computed only on `[BOS] text... [EOS]`

## Vocabulary

| Range | Contents | Size |
|---|---|---|
| 0 – text_vocab_size-1 | Hebrew chars, ASCII, punctuation, special tokens | ~120 |
| text_vocab_size – end | Kanade codec tokens (offset) | 12,800 |
| **Total** | | **~12,920** |

Special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, `[EOA]`

## Model

Qwen3 decoder initialized from scratch (~37M parameters):

| Hyperparameter | Value |
|---|---|
| `hidden_size` | 512 |
| `intermediate_size` | 1536 (3×) |
| `num_hidden_layers` | 8 |
| `num_attention_heads` | 4 |
| `num_key_value_heads` | 1 (MQA) |
| `head_dim` | 128 |
| `max_position_embeddings` | 2048 |
| Backbone params | ~30.8M |
| LM head params | ~6.7M |
| **Total** | **~37.4M** |

## Audio Tokenizer

[Kanade](https://huggingface.co/frothywater/kanade-25hz-clean) — 25Hz FSQ codec with codebook size 12,800 (levels `[8,8,8,5,5]`). Audio is encoded offline during dataset preparation; the Kanade model is not part of the ASR training graph.

## Data Pipeline

1. **`scripts/prepare.sh`** — downloads ILSpeech, runs `scripts/prepare_dataset.py`
2. **`scripts/prepare_dataset.py`** — for each sample: encode wav with Kanade → tokenize Hebrew text → write `{text_tokens, codec_tokens}` to JSONL
3. **`src/data.py`** — loads JSONL, builds sequences with codec offset + special tokens, pads batch

## Training

- Optimizer: AdamW with cosine schedule + warmup
- Mixed precision: fp16
- Multi-GPU: `accelerate launch`
- Entry point: `scripts/train.sh`

## Inference

```
wav → Kanade encode → codec_tokens → model.generate() → Hebrew text
```

Entry point: `uv run src/infer.py --checkpoint <dir> --wav <file>`
