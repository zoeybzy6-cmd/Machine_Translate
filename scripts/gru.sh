# python -u main.py --model gru --tf_mode mixed --mode train --epochs 15 --rnn_attn_type concat --emb_dim 512 --hid_dim 512 --dropout 0.5 --dataset_size 100k 2>&1 | tee -a ./logs/gru_mixed05_concat.log

python -u main.py --model gru --tf_mode force --mode test --epochs 15 --rnn_attn_type concat --emb_dim 512 --dropout 0.5 --hid_dim 512 --dataset_size 100k --decode_strategy beam 2>&1 | tee -a ./logs/gru_force_concat.log

# python -u main.py --model gru --tf_mode free --mode train --epochs 15 --rnn_attn_type concat --emb_dim 512 --hid_dim 512 --dropout 0.5 --dataset_size 100k 2>&1 | tee -a ./logs/gru_free_concat.log

# python -u main.py --model gru --tf_mode force --mode train --epochs 15 --rnn_attn_type dot --emb_dim 512 --hid_dim 512 --dropout 0.5 --dataset_size 100k 2>&1 | tee -a ./logs/gru_force_dot.log

# python -u main.py --model gru --tf_mode force --mode train --epochs 15 --rnn_attn_type general --emb_dim 512 --hid_dim 512 --dropout 0.5 --dataset_size 100k 2>&1 | tee -a ./logs/gru_force_general.log

