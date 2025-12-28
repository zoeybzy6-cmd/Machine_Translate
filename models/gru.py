import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layers=2, weights=None):
        super().__init__()
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=1)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=1)
        
        # Requirement: Consisting of two unidirectional layers 
        # bidirectional=False, num_layers=n_layers (default 2)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=n_layers, bidirectional=False, dropout=dropout)
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        
        # outputs: [src len, batch size, enc_hid_dim] (Unidirectional)
        # hidden: [n_layers, batch size, enc_hid_dim]
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, method='dot'):
        super().__init__()
        self.method = method # dot, general, concat
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        # Implementation of different alignment functions 
        if self.method == 'general': # Multiplicative: score(s_t, h_i) = s_t^T W h_i
            self.W = nn.Linear(enc_hid_dim, dec_hid_dim)
            
        elif self.method == 'concat': # Additive: score(s_t, h_i) = v^T tanh(W [s_t; h_i])
            self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
            
        # 'dot': Dot-product: score(s_t, h_i) = s_t^T h_i 
        # Requirement: enc_hid_dim must equal dec_hid_dim for dot product
        if self.method == 'dot':
            assert enc_hid_dim == dec_hid_dim, "For dot-product attention, encoder and decoder hidden dimensions must match."

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch size, dec_hid_dim] (Decoder state s_{t-1})
        # encoder_outputs: [src len, batch size, enc_hid_dim] (Encoder states h_i)
        # mask: [batch size, src len]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # Expand hidden: [batch, src_len, dec_hid]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # Permute encoder_outputs: [batch, src_len, enc_hid]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = None
        
        if self.method == 'dot':
            # Dot-product: sum(hidden * encoder_outputs) over dim 2
            # [batch, src_len, hid] * [batch, src_len, hid] -> sum -> [batch, src_len]
            energy = torch.sum(hidden_expanded * encoder_outputs, dim=2)
            
        elif self.method == 'general':
            # Multiplicative: hidden * W(encoder_outputs)
            # W(enc): [batch, src_len, dec_hid]
            energy = self.W(encoder_outputs)
            energy = torch.sum(hidden_expanded * energy, dim=2)
            
        elif self.method == 'concat':
            # Additive: v * tanh(W [hidden; enc])
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            energy = torch.tanh(self.attn(combined)) # [batch, src_len, dec_hid]
            energy = self.v(energy).squeeze(2)       # [batch, src_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        return F.softmax(energy, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, n_layers=2, weights=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=1)
        else:
            self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=1)
        
        # RNN input: embedding + context_vector
        # num_layers=n_layers (default 2, unidirectional) 
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, num_layers=n_layers, bidirectional=False, dropout=dropout)
        
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask=None):
        # input: [batch size]
        # hidden: [n_layers, batch size, dec_hid_dim]
        # encoder_outputs: [src len, batch size, enc_hid_dim]
        
        # print(f"Decoder input shape: {input.shape}, hidden shape: {hidden.shape}")
        # print(f"Encoder outputs shape: {encoder_outputs.shape}")
        input = input.unsqueeze(0) # [1, batch]
        embedded = self.dropout(self.embedding(input))
        
        # Calculate Attention using the top layer hidden state
        # hidden[-1] is the last layer's hidden state
        a = self.attention(hidden[-1], encoder_outputs, mask)
        # print(f"Attention weights shape: {a.shape}")
        a = a.unsqueeze(1) # [batch, 1, src len]
        
        # Weighted source vector
        encoder_outputs_permuted = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs_permuted) # [batch, 1, enc_hid]
        weighted = weighted.permute(1, 0, 2) # [1, batch, enc_hid]
        
        # RNN Input: [embedded; weighted]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # print(f"RNN input shape: {rnn_input.shape}")
        
        # RNN Step
        output, hidden = self.rnn(rnn_input, hidden)
        # print(f"RNN output shape: {output.shape}, new hidden shape: {hidden.shape}")
        
        # Prediction
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # print(f"Prediction shape: {prediction.shape}")
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
    
    def create_mask(self, src):
        # src: [src len, batch size]
        # mask: [batch size, src len]
        mask = (src != self.pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Encoder
        encoder_outputs, hidden = self.encoder(src)
        # print(f"Encoder outputs shape: {encoder_outputs.shape}, Encoder hidden shape: {hidden.shape}")

        mask = self.create_mask(src)
        # Pass encoder hidden directly to decoder (sizes match: 2 layers, unidirectional)
        input = trg[0,:] # <sos> token
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            # output: [batch size, output dim]
            # print(f"Time step {t}, Decoder output shape: {output.shape}")
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs