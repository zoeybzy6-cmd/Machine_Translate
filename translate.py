import torch
from custom_data import SOS_IDX, EOS_IDX
import torch.nn.functional as F

# --- Transformer Inference Wrapper ---
def translate_sentence_transformer(sentence, src_vocab, trg_vocab, model, device, max_len=200):
    model.eval()
    BOS_IDX, EOS_IDX = 2, 3

    src_indexes = src_vocab.encode(sentence)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device) # [1, seq_len]
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [BOS_IDX]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == EOS_IDX:
            break
    
    translated_sentence = trg_vocab.decode(trg_indexes)
    # print(f"translated_sentence={translated_sentence}")
    return translated_sentence


def translate_sentence_transformer_beam(
    sentence, src_vocab, trg_vocab, model, device,
    max_len=200, beam_width=4, length_penalty=0.6
):
    """
    Beam search decoding for Transformer.

    Args:
        sentence (str): source sentence
        src_vocab: source processor with encode()
        trg_vocab: target processor with decode() (optional)
        model: transformer model
        device: cuda/cpu
        max_len (int)
        beam_width (int): beam size
        length_penalty (float): alpha in length normalization (GNMT style)

    Returns:
        list[int]: decoded token ids (including BOS/EOS)
    """
    model.eval()
    BOS_IDX, EOS_IDX = 2, 3

    # --- Encode source ---
    src_indexes = src_vocab.encode(sentence)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # --- Beam initialization ---
    # each beam item: (token_ids, log_prob)
    beams = [([BOS_IDX], 0.0)]
    finished = []

    for step in range(max_len):
        new_beams = []

        # expand each beam
        for tokens, log_prob in beams:
            # if already ended, keep in finished
            if tokens[-1] == EOS_IDX:
                finished.append((tokens, log_prob))
                continue

            trg_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # output: [1, trg_len, vocab_size]
            next_token_logits = output[:, -1, :]                 # [1, vocab_size]
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)  # [1, vocab_size]

            # get top-k candidates
            topk_log_probs, topk_ids = torch.topk(next_token_log_probs, beam_width, dim=-1)

            for k in range(beam_width):
                next_id = topk_ids[0, k].item()
                next_log_prob = topk_log_probs[0, k].item()

                new_tokens = tokens + [next_id]
                new_score = log_prob + next_log_prob

                new_beams.append((new_tokens, new_score))

        # if nothing to expand
        if len(new_beams) == 0:
            break

        # --- Rank beams with length normalization ---
        def normalized_score(tokens, score):
            # GNMT length penalty
            lp = ((5 + len(tokens)) / 6) ** length_penalty
            return score / lp

        new_beams = sorted(
            new_beams,
            key=lambda x: normalized_score(x[0], x[1]),
            reverse=True
        )

        # keep top beam_width
        beams = new_beams[:beam_width]

        # early stop if all beams finished
        if all(b[0][-1] == EOS_IDX for b in beams):
            finished.extend(beams)
            break

    # --- Select best finished hypothesis ---
    if len(finished) > 0:
        finished = sorted(
            finished,
            key=lambda x: normalized_score(x[0], x[1]),
            reverse=True
        )
        best_tokens = finished[0][0]
    else:
        # no EOS found, take best unfinished
        best_tokens = beams[0][0]

    return trg_vocab.decode(best_tokens)

def translate_sentence_rnn(sentence, src_vocab, trg_vocab, model, device, max_len=200):
    """Single-sentence inference function of the RNN model"""
    model.eval()

    src_indexes = src_vocab.encode(sentence)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indexes = [SOS_IDX]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break
    
    translated_sentence = trg_vocab.decode(trg_indexes)
    return translated_sentence


# def translate_sentence_rnn_beam(
#     sentence,
#     src_processor,
#     tgt_processor,
#     model,
#     device,
#     max_len=50,
#     beam_size=5
# ):
#     model.eval()

#     # Encode
#     src_ids = src_processor.encode(sentence)
#     src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
#     # mask = model.create_mask(src_ids)

#     with torch.no_grad():
#         encoder_outputs, hidden = model.encoder(src_tensor)

#     beams = [([SOS_IDX], hidden, 0.0)]

#     for _ in range(max_len):
#         new_beams = []

#         for seq, h, score in beams:
#             if seq[-1] == EOS_IDX:
#                 new_beams.append((seq, h, score))
#                 continue

#             trg_tensor = torch.LongTensor([seq[-1]]).to(device)

#             with torch.no_grad():
#                 output, new_hidden = model.decoder(
#                     trg_tensor, h, encoder_outputs
#                 )

#             log_probs = torch.log_softmax(output.squeeze(0), dim=0)

#             topk = torch.topk(log_probs, beam_size)

#             for i in range(beam_size):
#                 token = topk.indices[i].item()
#                 token_score = topk.values[i].item()
#                 new_beams.append(
#                     (seq + [token], new_hidden, score + token_score)
#                 )

#         beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]

#     best_seq = beams[0][0]
#     return tgt_processor.decode(best_seq)


def translate_sentence_rnn_beam(
    sentence,
    src_processor,
    tgt_processor,
    model,
    device,
    max_len=50,
    beam_size=5,
    length_penalty=0.6
):
    model.eval()

    src_ids = src_processor.encode(sentence)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    beams = [([SOS_IDX], hidden, 0.0)]
    finished = []

    def normalized_score(seq, score):
        lp = ((5 + len(seq)) / 6) ** length_penalty
        return score / lp

    for _ in range(max_len):
        new_beams = []

        for seq, h, score in beams:
            if seq[-1] == EOS_IDX:
                finished.append((seq, h, score))
                continue

            trg_tensor = torch.LongTensor([seq[-1]]).to(device)

            with torch.no_grad():
                output, new_hidden = model.decoder(trg_tensor, h, encoder_outputs)

            log_probs = F.log_softmax(output.squeeze(0), dim=0)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                token = topk_ids[i].item()
                token_score = topk_log_probs[i].item()

                new_seq = seq + [token]
                new_score = score + token_score
                new_beams.append((new_seq, new_hidden, new_score))

        if len(new_beams) == 0:
            break

        new_beams = sorted(
            new_beams,
            key=lambda x: normalized_score(x[0], x[2]),
            reverse=True
        )
        beams = new_beams[:beam_size]

        if all(seq[-1] == EOS_IDX for seq, _, _ in beams):
            finished.extend(beams)
            break

    candidates = finished if len(finished) > 0 else beams
    best_seq = max(candidates, key=lambda x: normalized_score(x[0], x[2]))[0]

    return tgt_processor.decode(best_seq)
