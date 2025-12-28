export HF_ENDPOINT="https://hf-mirror.com"
python main.py --mode train --model t5 --dataset_size 100k --lr 3e-4 --epochs 15 2>&1 | tee ./logs/t5_log.txt