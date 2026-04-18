#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/infer.sh <checkpoint-dir> <wav-file>
#   ./scripts/infer.sh outputs/asr/step-500 path/to/audio.wav

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir> <wav-file>"}
WAV=${2:?"Usage: $0 <checkpoint-dir> <wav-file>"}

uv run src/infer.py --checkpoint "$CHECKPOINT" --wav "$WAV"
