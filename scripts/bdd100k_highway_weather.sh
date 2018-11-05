#!/usr/bin/env bash
train_mode=${1}
test_mode=${2}
epoch=10
input_height=256
input_width=256
output_height=$input_height
output_width=$input_width
batch_size=32


if [ $train_mode = "1" ]; then
	echo "Running ALOCC in TRAIN mode with bdd100k dataset"
	python3 train.py --dataset bdd100k --dataset_address /data/bdd100k/images/train_and_val_256by256 --input_height $input_height --input_width $input_width --output_height $output_height --output_width $output_width --epoch $epoch --batch_size $batch_size
fi

if [ $test_mode = "1" ]; then
	echo "Running ALOCC in TEST mode with bdd100k dataset"
	python3 test.py --dataset bdd100k --dataset_address /data/bdd100k/images/train_and_val_256by256 --input_height $input_height --input_width $input_width --output_height $output_height --output_width $output_width --epoch $epoch --batch_size $batch_size
fi
