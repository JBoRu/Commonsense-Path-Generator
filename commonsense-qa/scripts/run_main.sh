#!/bin/bash

python -u main.py \
	--dataset "csqa" \
	--inhouse 1 \
	--save_dir "./saved_models/csqa/" \
	--encoder 'roberta-large' \
	--max_seq_len 80 \
	--encoder_lr 2e-6 \
	--decoder_lr 1e-3 \
	--batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 7 \
	--nprocs 20 \
	--save_model 0 \
	--seed 42 \
