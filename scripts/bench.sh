#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir> [--save <report.txt>]"}
shift
EXTRA="$@"

uv run scripts/benchmark.py --checkpoint "$CHECKPOINT" --eval-dataset dataset/val.jsonl $EXTRA
