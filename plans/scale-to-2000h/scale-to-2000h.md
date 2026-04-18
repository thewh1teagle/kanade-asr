# Plan: Scale to 2000h

## Current state

- Model: ~10M params (4 layers, hidden 256) — validation model on ILSpeech (~1.5k samples)
- Goal: validate pipeline, confirm the causal codec→text approach works

## Phase 2: Scale up

Once pipeline is validated on small data, retrain from scratch on ~2000h Hebrew speech.

### Model config (target ~150M)

```python
Qwen3Config(
    hidden_size=1024,
    intermediate_size=1024 * 3,
    num_hidden_layers=16,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=128,
    max_position_embeddings=4096,
    vocab_size=text_vocab_size + audio_vocab_size,
)
```

### Data

- Target: ~2000h Hebrew speech with transcripts
- Possible sources: ILSpeech, Common Voice Hebrew, custom recordings
- Same pipeline: Kanade encode offline → JSONL

### Training changes

- Larger batch size (gradient accumulation)
- Longer warmup (~1000 steps)
- Consider curriculum: short utterances first, then longer
- Multi-GPU likely required

### Evaluation

- WER on ILSpeech test split
- Implement `scripts/benchmark.py` with WER metric
