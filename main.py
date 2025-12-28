import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from config import get_args
from custom_data import load_data_and_get_loaders, PAD_IDX
from utils import init_weights, count_parameters, epoch_time, calculate_bleu, evaluate_bleu_batch
from translate import translate_sentence_transformer, translate_sentence_transformer_beam, translate_sentence_rnn, translate_sentence_rnn_beam

# Import Models
from models.gru import Encoder as GRUEncoder, Decoder as GRUDecoder, Seq2Seq as GRUSeq2Seq, Attention
from models.transformer import Encoder as TransEncoder, Decoder as TransDecoder, Transformer
from models.t5 import train_t5, test_t5

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=2000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.step()

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        lr = lr * 0.5
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


def train_epoch(args, model, loader, optimizer, scheduler, criterion, clip, device, tf_ratio):
    """
    Generic training loop.
    tf_ratio: Probability of using teacher forcing (1.0 = Force, 0.0 = Free)
    """
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(loader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        # Transformer output needs adjusting: trg[:, :-1] as input
        if args.model == "transformer":
            output = model(src, trg[:, :-1]) # [batch, trg_len-1, output_dim]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
        else:
            # GRU
            # output shape: [trg_len, batch_size, output_dim]
            assert src.dim() == 2 and src.shape[1] == trg.shape[1], "GRU expects [len,batch]"
            output = model(src, trg, teacher_forcing_ratio=tf_ratio) 
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            if isinstance(model, Transformer):
                output = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
            else:
                output = model(src, trg, 0) # 0 = no teacher forcing
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

def main():
    args = get_args()
    set_seed(args.seed)
    
    # --- Checkpoint Directory Setup ---
    # Structure: logs/{model}/best_model.pt
    if args.model == 'gru':
        model_save_dir = os.path.join(args.ckpt_dir, args.model, args.tf_mode, args.rnn_attn_type)
    elif args.model == 'transformer':
        model_save_dir = os.path.join(args.ckpt_dir, args.model, args.pos_type, args.norm_type)
    else: # t5
        model_save_dir = os.path.join(args.ckpt_dir, args.model)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    # --- T5 Special Handling ---
    if args.model == 't5':
        # Align T5 save path logic with GRU/Transformer structure
        # Passing model_save_dir via args trick or modifying t5.py would be cleaner,
        # but for now we keep the interface consistent.
        # Note: You might need to adjust t5.py to accept a specific save dir if you want strict consistency.
        # Below assumes t5.py uses args.ckpt_dir.
        if args.mode == 'train':
            train_t5(args)
            test_t5(args)
        else:
            test_t5(args)
        return
    
    # --- Determine TF Ratio (Valid ONLY for RNN) ---
    current_tf_ratio = args.tf_ratio

    if args.model == 'transformer':
        # Force default behavior for Transformer to avoid confusion
        print(f">>> Model: TRANSFORMER (Teacher Forcing Mode is irrelevant/automatic)")
        current_tf_ratio = 0.0 # Placeholder, not used inside Transformer training loop
    else:
        # Logic for RNN (GRU)
        if args.tf_mode == 'force':
            current_tf_ratio = 1.0
            print(">>> Training Mode: TEACHER FORCING (Ratio = 1.0)")
        elif args.tf_mode == 'free':
            current_tf_ratio = 0.0
            print(">>> Training Mode: FREE RUNNING (Ratio = 0.0)")
            print("Warning: Pure Free Running from scratch is very hard to converge!")
        else:
            current_tf_ratio = args.tf_ratio
            print(f">>> Training Mode: MIXED (Ratio = {current_tf_ratio})")

    # --- Standard PyTorch Workflow (GRU / Transformer) ---
    print(f"Loading Data (Size: {args.dataset_size})...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_first = True if args.model == 'transformer' else False
    
    train_path = os.path.join(args.data_dir, f'train_{args.dataset_size}.jsonl')
    valid_path = os.path.join(args.data_dir, 'valid.jsonl')
    test_path = os.path.join(args.data_dir, 'test.jsonl')
    
    train_ds, train_ld, valid_ds, valid_ld, test_ds, test_ld, src_processor, tgt_processor = load_data_and_get_loaders(
        train_path, valid_path, test_path, 
        batch_size=args.batch_size, batch_first=batch_first
    )

    # --- Model Initialization ---
    input_dim = len(src_processor)
    output_dim = len(tgt_processor)

    print(f"Initializing {args.model.upper()} model...")
    print(f"Config: Emb={args.emb_dim}, Hid={args.hid_dim}, Layers={args.n_layers}")
    print(f"Vocab Sizes: Src(ch)={input_dim}, Trg(en)={output_dim}")

    print(f"Initializing {args.model.upper()} model with Pre-trained Embeddings...")

    if args.model == 'transformer':
        enc = TransEncoder(input_dim, args.hid_dim, args.n_layers, args.n_heads, 
                           args.pf_dim, args.dropout, device, PAD_IDX, norm_type=args.norm_type, pos_type=args.pos_type)
        dec = TransDecoder(output_dim, args.hid_dim, args.n_layers, args.n_heads, 
                           args.pf_dim, args.dropout, device, PAD_IDX, norm_type=args.norm_type, pos_type=args.pos_type)
        model = Transformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
        

        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
        model.apply(initialize_weights)

    else:
        # RNN Initialization
        print(f"RNN Settings: Layers={args.n_layers}, Attention={args.rnn_attn_type}")
        
        # Initialize Attention based on type 
        attn = Attention(args.hid_dim, args.hid_dim, method=args.rnn_attn_type)
        
        # Initialize Encoder/Decoder with n_layers 
        enc = GRUEncoder(input_dim, args.emb_dim, args.hid_dim, args.hid_dim, args.dropout, n_layers=args.n_layers)
        dec = GRUDecoder(output_dim, args.emb_dim, args.hid_dim, args.hid_dim, args.dropout, attn, n_layers=args.n_layers)
        
        model = GRUSeq2Seq(enc, dec, device).to(device)
        model.apply(init_weights)

    # Define Best Model Path
    save_path = os.path.join(model_save_dir, 'best_model.pt')
    
    
    if args.model == 'transformer':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-7,                 # Noam/WarmupSchedule 0.0
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        scheduler = WarmupScheduler(optimizer, args.hid_dim, args.warmup_steps)

    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    
    try:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # --- Training Loop ---
    if args.mode == 'train':
        print(f"Starting Training {args.model.upper()}...")
        print(f'Trainable parameters: {count_parameters(model)}')
        
        best_valid_loss = float('inf')

        for epoch in range(args.epochs):
            start_time = time.time()

            if args.model == 'gru' and args.tf_mode == 'mixed':
                decay_start = 5
                decay_end = 15
                if epoch < decay_start:
                    current_tf_ratio = 1.0
                elif epoch > decay_end:
                    current_tf_ratio = 0.0
                else:
                    # Linearly decrease from 1.0 to 0.0
                    current_tf_ratio = 1.0 - ((epoch - decay_start) / (decay_end - decay_start))
                print(f"[Epoch {epoch+1}] TF Ratio: {current_tf_ratio:.2f}")

            # Pass the dynamic current_tf_ratio to train_epoch
            train_loss = train_epoch(args, model, train_ld, optimizer, scheduler, criterion, args.clip, device, current_tf_ratio)
            valid_loss = evaluate(model, valid_ld, criterion, device)

            val_bleu = evaluate_bleu_batch(
                model=model,
                data_loader=valid_ld,
                src_proc=src_processor,
                tgt_proc=tgt_processor,
                device=device,
                max_len=200,
                is_transformer=(args.model == "transformer")
            )
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'epoch_{epoch+1}.pt'))
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), save_path)
                print(f"--> Best model saved to {save_path}")
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} | PPL: {math.exp(valid_loss):7.3f}')
            print(f"\t Val BLEU: {val_bleu:.2f}\n")

    # --- Testing ---
    # save_path = os.path.join(model_save_dir, 'epoch_15.pt')
    print(f"Loading best model for testing from {save_path} ...")
    
    if os.path.exists(save_path):
        # Use weights_only=True to suppress warning if using newer PyTorch
        try:
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        except TypeError:
            # Fallback for older PyTorch versions
            model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        # !!! Error Handling for Test Mode !!!
        raise FileNotFoundError(f"Error: Best model weight file not found at '{save_path}'. "
                                f"Please train the model first or check the --ckpt_dir path.")

    test_loss = evaluate(model, test_ld, criterion, device)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    # --- BLEU Evaluation with Decoding Policy  ---
    if args.model == 'transformer':
        if args.decode_strategy == 'beam':
            translate_func = lambda s, s_v, t_v, m, d, ml: translate_sentence_transformer_beam(
                s, s_v, t_v, m, d, max_len=ml,
                beam_width=args.beam_width,
                length_penalty=args.length_penalty
            )
        else:
            translate_func = translate_sentence_transformer
    else:
        # Select decoding function based on strategy
        print(f"Evaluating BLEU with strategy: {args.decode_strategy.upper()}")
        if args.decode_strategy == 'beam':
                translate_func = lambda s, s_v, t_v, m, d, ml: translate_sentence_rnn_beam(s, s_v, t_v, m, d, ml, args.beam_width, length_penalty=args.length_penalty)
        else:
            translate_func = translate_sentence_rnn
    
    start_time = time.time()
    score = calculate_bleu(test_ds.data, src_processor, tgt_processor, model, device, translate_func)
    end_time = time.time()
    total_time = end_time - start_time
    num_sent = len(test_ds.data)
    avg_time = total_time / num_sent

    print(f'BLEU score = {score*100:.2f}')
    print(f"Decoding strategy = {args.decode_strategy.upper()} (beam_width={args.beam_width if args.decode_strategy=='beam' else 1})")
    print(f"Total inference time = {total_time:.2f}s  | Avg time per sentence = {avg_time*1000:.2f} ms")


if __name__ == '__main__':
    main()