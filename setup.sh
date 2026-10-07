#!/bin/bash
# HIVE-Bench one-shot setup
# Run from project root: bash setup.sh

echo "[HIVE-Bench] Installing hivebench policy framework..."
pip install -e . --quiet

echo "[HIVE-Bench] Done. You can now run:"
echo "  bash Bench/Robocasa_tabletop/train_files/run_vision_robocasa.sh"
