import torch
from custom_data import SOS_IDX, EOS_IDX

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


def translate_sentence_rnn_beam(
    sentence,
    src_processor,
    tgt_processor,
    model,
    device,
    max_len=50,
    beam_size=5
):
    model.eval()

    # Encode
    src_ids = src_processor.encode(sentence)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    # mask = model.create_mask(src_ids)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    beams = [([SOS_IDX], hidden, 0.0)]

    for _ in range(max_len):
        new_beams = []

        for seq, h, score in beams:
            if seq[-1] == EOS_IDX:
                new_beams.append((seq, h, score))
                continue

            trg_tensor = torch.LongTensor([seq[-1]]).to(device)

            with torch.no_grad():
                output, new_hidden = model.decoder(
                    trg_tensor, h, encoder_outputs
                )

            log_probs = torch.log_softmax(output.squeeze(0), dim=0)

            topk = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                token = topk.indices[i].item()
                token_score = topk.values[i].item()
                new_beams.append(
                    (seq + [token], new_hidden, score + token_score)
                )

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]

    best_seq = beams[0][0]
    return tgt_processor.decode(best_seq)