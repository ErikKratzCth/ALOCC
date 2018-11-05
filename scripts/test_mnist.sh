#!/usr/bin/env bash
checkpoint_dir="checkpoint/mnist_32_28_28"
python3 test.py --dataset mnist --dataset_address ./dataset/mnist/ --input_height 28 --output_height 28 --checkpoint_dir $checkpoint_dir 


