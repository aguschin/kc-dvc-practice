#!/bin/sh
python scripts/prepare.py
python scripts/split.py
python scripts/train.py
python scripts/evaluate.py
