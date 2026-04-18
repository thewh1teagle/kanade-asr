# kanade-asr

Hebrew ASR using [Kanade](https://github.com/frothywater/kanade-tokenizer) audio codec tokens + Qwen3 causal LM (~10M params).

## Setup

```console
uv sync
```

## Prepare data

```console
bash scripts/prepare.sh
```

## Train

```console
bash scripts/train.sh
```

Resume from checkpoint:

```console
bash scripts/train.sh --resume outputs/asr/step-1000
```

## Inference

```console
bash scripts/infer.sh outputs/asr/step-1000 path/to/audio.wav
```
