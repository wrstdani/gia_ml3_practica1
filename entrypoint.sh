#!/bin/bash
set -euo pipefail

if [ $# -ge 2 ]; then
    TRAIN_CSV="$1"
    TEST_CSV="$2"
else
    TRAIN_CSV="${TRAIN_CSV:-}"
    TEST_CSV="${TEST_CSV:-}"
fi

if [ -z "TRAIN_CSV" ] || [ -z "$TEST_CSV" ]; then
    echo "Uso: docker run ... <train.csv> <test.csv> OR docker run -e TRAIN_CSV=/path/to/train.csv -e TEST_CSV=/path/to/test.csv"
    exit 1
fi

exec /gia_ml3_practica1/venv/bin/python -u /gia_ml3_practica1/src/mixed_manifold_detector.py "$TRAIN_CSV" "$TEST_CSV"
