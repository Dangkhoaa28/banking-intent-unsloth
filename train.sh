#!/bin/bash

# 1. Preprocess data
echo "Preprocessing data..."
python scripts/preprocess_data.py

# 2. Train model
echo "Starting training..."
python scripts/train.py
