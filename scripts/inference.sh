#!/bin/bash
# python inference.py --model gru --dataset_size 100k

python inference.py --model transformer --dataset_size 100k --hid_dim 512 --n_layers 3 --n_heads 8 --pf_dim 2048 --pos_type absolute --norm_type rmsnorm

# python inference.py --model t5 --dataset_size 100k