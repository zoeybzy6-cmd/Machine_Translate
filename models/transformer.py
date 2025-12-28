import torch
import torch.nn as nn


# =========================================================
# 1) Norm Layers
# =========================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.scale


def build_norm(norm_type, hid_dim):
    if norm_type == "layernorm":
        return nn.LayerNorm(hid_dim)
    elif norm_type == "rmsnorm":
        return RMSNorm(hid_dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# =========================================================
# 2) ALiBi Helpers (Relative Pos Bias)
# =========================================================

def get_alibi_slopes(n_heads):
    """
    Official ALiBi uses a special slope rule.
    For ablation, a simplified slope is OK and still works well.
    """
    return torch.linspace(1.0, 0.1, steps=n_heads)


def build_alibi_bias(n_heads, q_len, k_len, device):
    """
    Return ALiBi bias: shape [1, n_heads, q_len, k_len]
    """
    slopes = get_alibi_slopes(n_heads).to(device)              # [h]
    q_pos = torch.arange(q_len, device=device).unsqueeze(1)    # [q,1]
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)    # [1,k]
    dist = (k_pos - q_pos).abs().float()                       # [q,k]
    bias = -dist.unsqueeze(0) * slopes.view(n_heads, 1, 1)     # [h,q,k]
    return bias.unsqueeze(0)                                   # [1,h,q,k]


# =========================================================
# 3) Transformer Wrapper
# =========================================================

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src: [batch, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, src_len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg: [batch, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, trg_len]

        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # [batch, 1, trg_len, trg_len]
        return trg_mask

    def forward(self, src, trg):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output


# =========================================================
# 4) Encoder
# =========================================================

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        pad_idx,
        weights=None,
        max_length=200,
        norm_type="layernorm",
        pos_type="absolute",
    ):
        super().__init__()

        if weights is not None:
            self.tok_embedding = nn.Embedding.from_pretrained(
                weights, freeze=False, padding_idx=pad_idx
            )
        else:
            self.tok_embedding = nn.Embedding(input_dim, hid_dim, padding_idx=pad_idx)

        self.device = device
        self.pos_type = pos_type
        self.use_alibi = (pos_type == "alibi")

        # absolute pos encoding only
        if pos_type == "absolute":
            self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length)
        else:
            self.pos_embedding = None

        self.layers = nn.ModuleList([
            EncoderLayer(
                hid_dim, n_heads, pf_dim, dropout, device,
                norm_type=norm_type,
                use_alibi=self.use_alibi
            )
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src: [batch, src_len]
        x = self.tok_embedding(src) * self.scale
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        pf_dim,
        dropout,
        device,
        norm_type="layernorm",
        use_alibi=False
    ):
        super().__init__()
        self.self_attn_layer_norm = build_norm(norm_type, hid_dim)
        self.ff_layer_norm = build_norm(norm_type, hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)
        self.use_alibi = use_alibi

    def forward(self, src, src_mask):
        # self-attn
        _src, _ = self.self_attention(src, src, src, src_mask, use_alibi=self.use_alibi)
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # ff
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


# =========================================================
# 5) Multi-Head Attention
# =========================================================

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None, use_alibi=False):
        batch_size = query.size(0)

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # [batch, heads, len, head_dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # energy: [batch, heads, q_len, k_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if use_alibi:
            bias = build_alibi_bias(self.n_heads, Q.size(2), K.size(2), Q.device)
            energy = energy + bias

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)

        # [batch, q_len, hid_dim]
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention


# =========================================================
# 6) Feedforward + Positional Encoding
# =========================================================

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hid_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        P = torch.zeros((1, max_len, hid_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, hid_dim, 2, dtype=torch.float32) / hid_dim
        )
        P[0, :, 0::2] = torch.sin(X)
        P[0, :, 1::2] = torch.cos(X)
        self.register_buffer("P", P)

    def forward(self, x):
        x = x + self.P[:, :x.size(1), :]
        return self.dropout(x)


# =========================================================
# 7) Decoder
# =========================================================

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        pad_idx,
        weights=None,
        max_length=200,
        norm_type="layernorm",
        pos_type="absolute",
    ):
        super().__init__()

        if weights is not None:
            self.tok_embedding = nn.Embedding.from_pretrained(
                weights, freeze=False, padding_idx=pad_idx
            )
        else:
            self.tok_embedding = nn.Embedding(output_dim, hid_dim, padding_idx=pad_idx)

        self.device = device
        self.pos_type = pos_type
        self.use_alibi = (pos_type == "alibi")

        if pos_type == "absolute":
            self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length)
        else:
            self.pos_embedding = None

        self.layers = nn.ModuleList([
            DecoderLayer(
                hid_dim, n_heads, pf_dim, dropout, device,
                norm_type=norm_type,
                use_alibi=self.use_alibi
            )
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch, trg_len]
        x = self.tok_embedding(trg) * self.scale
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)

        attention = None
        for layer in self.layers:
            x, attention = layer(x, enc_src, trg_mask, src_mask)

        output = self.fc_out(x)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        pf_dim,
        dropout,
        device,
        norm_type="layernorm",
        use_alibi=False
    ):
        super().__init__()
        self.self_attn_layer_norm = build_norm(norm_type, hid_dim)
        self.enc_attn_layer_norm = build_norm(norm_type, hid_dim)
        self.ff_layer_norm = build_norm(norm_type, hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)
        self.use_alibi = use_alibi

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # self-attn (use alibi optionally)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask, use_alibi=self.use_alibi)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # enc-attn (usually no alibi here)
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask, use_alibi=False)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # ff
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention
