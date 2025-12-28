import argparse

def get_args():
    parser = argparse.ArgumentParser(description='NMT Project: GRU, Transformer, and T5')

    # --- Mode Selection ---
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Execution mode: "train" (trains then tests) or "test" (loads best model and tests)')
    
    # --- Model Selection ---
    parser.add_argument('--model', type=str, default='gru', choices=['gru', 'transformer', 't5'],
                        help='Model architecture to use')
    
    # --- Data Settings ---
    parser.add_argument('--dataset_size', type=str, default='10k', choices=['10k', '100k'],
                        help='Size of the training dataset (10k or 100k)')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing jsonl files')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Warm-up steps for learning rate scheduler')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    # --- RNN Specific Requirements ---
    # 1. Attention Mechanism Variations
    parser.add_argument('--rnn_attn_type', type=str, default='concat', choices=['dot', 'general', 'concat'],
                        help='Attention mechanism: dot (Dot-product), general (Multiplicative), concat (Additive)')
    
    # 2. Decoding Policy
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'beam'],
                        help='Decoding strategy for testing')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument("--length_penalty", type=float, default=0.6)

    # 3. Training Policy (Teacher Forcing vs Free Running)
    parser.add_argument('--tf_mode', type=str, default='mixed', choices=['force', 'free', 'mixed'],
                        help='Training Strategy')
    parser.add_argument('--tf_ratio', type=float, default=0.5)
    
    # --- Model Architecture Config (For GRU/Transformer) ---
    # Assignment requires 2 layers for RNN 
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers (Must be 2 for RNN assignment)')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Transformer specific args...
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--pf_dim', type=int, default=2048)
    parser.add_argument('--pos_type', type=str, default='absolute', choices=['absolute', 'alibi'], help='absolute pos enc vs relative ALiBi bias')
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'], help='layernorm vs rmsnorm')

    
    # --- Checkpoints ---
    parser.add_argument('--ckpt_dir', type=str, default='ckpts', help='Directory to save/load models')
    parser.add_argument('--model_path', type=str, default=None, help='Custom model checkpoint path for testing/inference')
    
    return parser.parse_args()