#!/usr/bin/env bash
train_mode=${1}
test_mode=${2}
epoch=50
input_height=256
input_width=256
output_height=$input_height
output_width=$input_width
batch_size=64
z_dim=256
gf_dim=64
df_dim=64


if [ $train_mode = "1" ]; then
	echo "Running ALOCC in TRAIN mode with bdd100k dataset"
	python3 train.py --dataset bdd100k --dataset_address /data/bdd100k/images/train_and_val_256by256 --input_height $input_height --input_width $input_width --output_height $output_height --output_width $output_width --epoch $epoch --batch_size $batch_size --z_dim $z_dim --gf_dim $gf_dim --df_dim $df_dim
fi

if [ $test_mode = "1" ]; then
	echo "Running ALOCC in TEST mode with bdd100k dataset"
	python3 test.py --dataset bdd100k --dataset_address /data/bdd100k/images/train_and_val_256by256 --input_height $input_height --input_width $input_width --output_height $output_height --output_width $output_width --epoch $epoch --batch_size $batch_size --z_dim $z_dim --gf_dim $gf_dim --df_dim $df_dim

fi
