import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from nltk.translate.bleu_score import corpus_bleu
except ImportError:
    corpus_bleu = None

from nltk.translate.bleu_score import SmoothingFunction
smooth = SmoothingFunction().method1

import numpy as np


UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

def init_weights(m):
    for name, param in m.named_parameters():
        if 'embedding' in name:
            continue
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.2f}B"
    elif params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M"
    else:
        return f"{params / 1_000:.2f}K"

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_vocab_sample(proc, n=20):
    print(f"\n=== {proc.lang.upper()} vocab sample (word2idx) ===")
    for i, (w, idx) in enumerate(proc.word2idx.items()):
        print(f"{i:3d}: {w} -> {idx}")
        if i >= n - 1:
            break

def calculate_bleu(
    data,
    src_proc,
    tgt_proc,
    model,
    device,
    translate_func,
    max_len=50
):
    """
    BLEU calculation for word-level NMT
    Compatible with LanguageProcessor.decode() -> string
    """
    smooth = SmoothingFunction().method4

    references = []   # List[List[List[str]]]
    hypotheses = []   # List[List[str]]

    print(f"Calculating BLEU on {len(data)} samples...")

    for idx, (src_text, tgt_text) in enumerate(data):
        # -------- 1. Model prediction (string) --------
        pred_sentence = translate_func(
            src_text, src_proc, tgt_proc, model, device, max_len
        )
        # -------- 2. Tokenize prediction --------
        pred_tokens = pred_sentence.strip().split()

        # -------- 3. Tokenize reference --------
        ref_tokens = tgt_text.strip().split()

        references.append([ref_tokens])
        hypotheses.append(pred_tokens)

        # -------- Debug --------
        if idx < 5:
            print(f"\n--- Sample {idx} ---")
            print(f"SRC: {src_text}")
            print(f"REF: {' '.join(ref_tokens)}")
            print(f"HYP: {' '.join(pred_tokens)}")

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} sentences")

    bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    return bleu


@torch.no_grad()
def greedy_decode_transformer_batch(model, src, src_pad_idx, max_len, sos_idx, eos_idx, device):
    """
    Greedy decode for a batch of src sentences.
    Args:
        model: Transformer model
        src: [batch, src_len]
        Returns:
        decoded_ids: [batch, <=max_len]
    """
    model.eval()
    batch_size = src.size(0)

    src = src.to(device)
    src_mask = model.make_src_mask(src)

    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)

    # Start with <sos>
    trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        trg_mask = model.make_trg_mask(trg)
        output, _ = model.decoder(trg, enc_src, trg_mask, src_mask)
        next_token = output[:, -1, :].argmax(dim=-1)   # [batch]

        trg = torch.cat([trg, next_token.unsqueeze(1)], dim=1)

        finished |= (next_token == eos_idx)
        if finished.all():
            break

    return trg



@torch.no_grad()
def greedy_decode_rnn_batch(
    model,
    src,
    sos_idx,
    eos_idx,
    pad_idx, 
    device,
    max_len=200,
):
    """
    Batch greedy decoding for your GRU Seq2Seq model.
    Assumes:
      - src shape: [src_len, batch]   (because your RNN uses batch_first=False)
      - model.encoder(src) returns (encoder_outputs, hidden)
      - model.decoder(trg_token, hidden, encoder_outputs) returns (output_logits, hidden)
        where trg_token shape is [batch] or [1,batch] depending on implementation.

    Returns:
      pred_ids: List[List[int]] length=batch, each is predicted token sequence including SOS/EOS
    """
    model.eval()

    src = src.to(device)  # [src_len, batch]
    src_len, batch_size = src.shape

    # --- Encode ---
    encoder_outputs, hidden = model.encoder(src)  # encoder_outputs: [src_len, batch, hid*?], hidden: [n_layers, batch, hid]
    mask = (src != pad_idx).permute(1, 0)  # [batch, src_len]

    # --- Init ---
    trg_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)  # [batch]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # store predicted tokens for each batch item
    pred_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)  # [batch, 1]

    for _ in range(max_len):
        # decoder expects token input shape maybe [batch] or [1,batch]; adjust if needed
        output, hidden = model.decoder(trg_token, hidden, encoder_outputs)  
        # output: [batch, vocab]
        next_token = output.argmax(dim=-1)  # [batch]

        # append
        pred_tokens = torch.cat([pred_tokens, next_token.unsqueeze(1)], dim=1)

        # update finished
        finished |= (next_token == eos_idx)
        if finished.all():
            break

        # next input
        trg_token = next_token

    return pred_tokens  # [batch, <=max_len+1]


@torch.no_grad()
def evaluate_bleu_batch(
    model,
    data_loader,
    src_proc,
    tgt_proc,
    device,
    max_len=200,
    is_transformer=True
):
    model.eval()

    sp_trg = getattr(tgt_proc, "sp", None)
    assert sp_trg is not None, "tgt_proc must have SentencePieceProcessor (.sp)"

    refs_text = []
    hyps_text = []

    refs_pieces = []
    hyps_pieces = []

    special = {tgt_proc.PAD_IDX, tgt_proc.SOS_IDX, tgt_proc.EOS_IDX}
    print(f"special token IDs: {special}")

    for src, trg in data_loader:
        src = src.to(device)
        trg = trg.to(device)

        if is_transformer:
            pred_ids = greedy_decode_transformer_batch(
                model=model,
                src=src,
                src_pad_idx=src_proc.PAD_IDX,
                max_len=max_len,
                sos_idx=tgt_proc.SOS_IDX,
                eos_idx=tgt_proc.EOS_IDX,
                device=device
            )

            pred_ids_cpu = pred_ids.detach().cpu().tolist()
            trg_cpu = trg.detach().cpu().tolist()

            for pred_seq, ref_seq in zip(pred_ids_cpu, trg_cpu):

                # ----- TEXT (decode directly from ids) -----
                hyp_t = tgt_proc.decode(pred_seq)
                ref_t = tgt_proc.decode(ref_seq)
                hyps_text.append(hyp_t)
                refs_text.append(ref_t)

                # ----- PIECES (convert ids -> pieces, no decode->encode) -----
                hyp_piece = [sp_trg.id_to_piece(i) for i in pred_seq if i not in special]
                ref_piece = [sp_trg.id_to_piece(i) for i in ref_seq if i not in special]
                hyps_pieces.append(hyp_piece)
                refs_pieces.append(ref_piece)

        else:
                # src: [src_len, batch], trg: [trg_len, batch]
            pred_ids = greedy_decode_rnn_batch(
                model=model,
                src=src,  # 这里 src 是已经 to(device) 的 [src_len,batch]
                sos_idx=tgt_proc.SOS_IDX,
                eos_idx=tgt_proc.EOS_IDX,
                pad_idx=tgt_proc.PAD_IDX,
                device=device,
                max_len=max_len
            )

            pred_ids_cpu = pred_ids.detach().cpu().tolist()
            trg_cpu = trg.detach().cpu().tolist()  # trg: [trg_len, batch] -> list of len trg_len

            # transpose trg_cpu to [batch, trg_len]
            trg_cpu = list(map(list, zip(*trg_cpu)))

            special = {tgt_proc.PAD_IDX, tgt_proc.SOS_IDX, tgt_proc.EOS_IDX}

            for pred_seq, ref_seq in zip(pred_ids_cpu, trg_cpu):
                hyp_text = tgt_proc.decode(pred_seq)
                ref_text = tgt_proc.decode(ref_seq)

                hyps_text.append(hyp_text)
                refs_text.append(ref_text)

                hyp_piece = [sp_trg.id_to_piece(i) for i in pred_seq if i not in special]
                ref_piece = [sp_trg.id_to_piece(i) for i in ref_seq if i not in special]

                hyps_pieces.append(hyp_piece)
                refs_pieces.append(ref_piece)

    refs = [[r] for r in refs_pieces]
    hyps = hyps_pieces
    return corpus_bleu(refs, hyps, smoothing_function=smooth) * 100