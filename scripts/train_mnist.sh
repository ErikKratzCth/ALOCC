#!/usr/bin/env bash
epoch=$1
python3 train.py --dataset mnist --dataset_address ./dataset/mnist/ --input_height 28 --output_height 28 --epoch $epoch

