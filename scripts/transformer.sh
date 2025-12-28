#!/bin/bash

# python -u main.py --mode train --model transformer --dataset_size 100k --hid_dim 512 --n_layers 3 --n_heads 8 --pf_dim 2048 --epochs 15 --pos_type absolute --norm_type layernorm \
#     2>&1 | tee -a ./logs/transformer_abs_layernorm.txt

python -u main.py --mode train --model transformer --dataset_size 100k --hid_dim 512 --n_layers 3 --n_heads 8 --pf_dim 2048 --epochs 15 --pos_type absolute --norm_type rmsnorm  \
    2>&1 | tee -a ./logs/transformer_abs_rmsnorm.txt

# python -u main.py --mode train --model transformer --dataset_size 100k --hid_dim 512 --n_layers 3 --n_heads 8 --pf_dim 2048 --epochs 15 --pos_type alibi --norm_type layernorm \
#     2>&1 | tee -a ./logs/transformer_alibi_layernorm.txt

# python -u main.py --mode train --model transformer --dataset_size 100k --hid_dim 512 --n_layers 3 --n_heads 8 --pf_dim 2048 --epochs 15 --pos_type alibi --norm_type rmsnorm \
#     2>&1 | tee -a ./logs/transformer_alibi_rmsnorm.txt