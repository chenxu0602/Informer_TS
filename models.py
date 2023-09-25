import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import math, copy, pyprind
import numpy as np

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class ProbSparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, topk_ratio=0.25):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.topk = int(self.d_k * topk_ratio)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return x.view(x.size(0), -1, self.n_heads, self.d_k)
    
    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_scores = torch.einsum("nqhd,nkhd->nhqk", [Q, K]) / (self.d_k ** 0.5)
        if mask is not None: attn_scores += (mask * -1e9)

        top_val, top_idx = attn_scores.topk(self.topk, sorted=False, dim=-1)
        attn_scores_topk = torch.zeros_like(attn_scores)
        attn_scores.scatter(dim=-1, index=top_idx, src=top_val)

        weights = torch.softmax(attn_scores, dim=-1)
        attn = torch.einsum("nhqk,nkhd->nqhd", [weights, V]).contiguous.view(query.size(0), -1, self.d_model)

        return self.W_o(attn), weights



class ProbSparseMultiheadAttention_2020(nn.Module):
    def __init__(self, d_model, n_heads, topk=5):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.topk = topk

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return x.view(x.size(0), -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

    def expand_mask(self, mask, batch_size, index):
        mask_ex = mask[None, None, :].expand(batch_size, self.n_heads, *mask.size())
        return mask_ex[torch.arange(batch_size)[:, None, None],
                       torch.arange(self.n_heads)[None, :, None],
                       index, :]
        

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        
        # attn_scores = torch.einsum("nhqd,nhkd->nhqk", [Q, K]) / math.sqrt(self.d_k)
        # if mask is not None: attn_scores += (mask * -1e9)

        L_Q = Q.size(2)
        L_K = K.size(2)
        log_L_Q = np.ceil(np.log1p(L_Q)).astype('int').item()
        log_L_K = np.ceil(np.log1p(L_K)).astype('int').item()

        U_part = min(self.topk * L_Q * log_L_K, L_K)

        index_sample = torch.randint(0, L_K, (U_part,))

        K_sample = K[:, :, index_sample, :]
        Q_K_sample = torch.einsum("nhqd,nhkd->nhqk", [Q, K])

        M = Q_K_sample.max(dim=-1)[0] - torch.div(Q_K_sample.sum(dim=-1), L_K)

        u = min(self.topk * log_L_Q, L_Q)

        _, M_top = M.topk(u, sorted=False)

        dim_for_slice_0 = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1)
        dim_for_slice_1 = torch.arange(self.n_heads).unsqueeze(0).unsqueeze(-1)

        Q_reduced = Q[dim_for_slice_0, dim_for_slice_1, M_top, :]

        attn_scores = torch.einsum("nhqd,nhkd->nhqk", [Q_reduced, K]) / math.sqrt(self.d_k)

        if mask is None: mask = torch.triu(torch.ones(L_Q, L_K), diagonal=1)
        if mask.dim() == 2: mask = self.expand_mask(mask, batch_size, M_top)

        attn_scores += (mask * -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn = torch.einsum("nhql,nhld->nqhd", [attn_probs, V]).contiguous().view(batch_size, -1, self.d_model)

        return attn, attn_probs


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_production_attention(self, Q, K, V, mask=None):
        attn_scores = torch.einsum("nqhd,nkhd->nhqk", [Q, K]) / math.sqrt(self.d_k)
        if mask is not None: attn_scores += (mask * -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn = torch.einsum("nhql,nlhd->nqhd", [attn_probs, V]).contiguous().view(Q.size(0), -1, self.d_model)
        return attn, attn_probs

    def split_heads(self, x):
        return x.view(x.size(0), -1, self.n_heads, self.d_k)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn, weights = self.scaled_dot_production_attention(Q, K, V, mask)
        return self.W_o(attn), weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ConvLayer(nn.Module):
    def __init__(self, d_model, window_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=window_size,
            padding=2,
            padding_mode='circular',
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1d(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)
        
# class InformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attn = ProbSparseMultiheadAttention(d_model, n_heads)
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, mask=None):
#         attn, _ = self.self_attn(x, x, x, mask)
#         x = self.norm1(x + self.dropout(attn))
#         ff = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
#         x = self.norm2(x + self.dropout(self.conv2(ff).transpose(-1, 1)))
#         return x

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseMultiheadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            nn.ReLU(),
            nn.Conv1d(d_ff, d_model, 1)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn, weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))

        ffn = self.feed_forward(x.transpose(1, 2))
        return self.norm2(x + self.dropout(ffn).transpose(1, 2)), weights


# class InformerEncoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super().__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer

#     def forward(self, x, mask=None):
#         attns = []
#         if self.conv_layers is not None:
#             for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
#                 x, weights = attn_layer(x, mask=mask)
#                 x = conv_layer(x)
#                 attns += weights,

#             x, weights = self.attn_layers[-1](x, mask=mask)
#             attns += weights,
#         else:
#             for attn_layer in self.attn_layers:
#                 x, weights = attn_layer(x, mask=mask)
#                 attns += weights,

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, attns

class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, n_layers=3):
        super().__init__()
        self.attn_layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.conv_layers = nn.ModuleList([ConvLayer(d_model) for _ in range(n_layers - 1)])
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x, mask=None):
        attns = []
        for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            x, weights = attn_layer(x, mask=mask)
            x = conv_layer(x)
            attns += weights,

        x, weights = self.attn_layers[-1](x, mask=mask)
        attns += weights,

        return self.norm(x), attns

# class InformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attn = ProbSparseMultiheadAttention(d_model, n_heads)
#         self.cross_attn = MultiheadAttention(d_model, n_heads)
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, cross, x_mask, cross_mask=None):
#         attn, _ = self.self_attn(x, x, x, x_mask)
#         x = self.norm1(x + self.dropout(attn))
#         attn, _ = self.cross_attn(x, cross, cross, cross_mask)
#         x = self.norm2(x + self.dropout(attn))
#         ff = self.dropout(self.activation(self.conv1(ff.transpose(-1, 1))))
#         ff = self.dropout(self.conv2(ff).transpose(-1, 1))
#         x = self.norm3(x + ff)
#         return x

class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseMultiheadAttention(d_model, n_heads)
        self.cross_attn = ProbSparseMultiheadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            nn.ReLU(),
            nn.Conv1d(d_ff, d_model, 1)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, x_mask, memory_mask):
        attn_x, weights_x = self.self_attn(x, x, x, x_mask)
        x = self.norm1(x + self.dropout(attn_x))
        attn_memory, weights_memory = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(attn_memory))
        ffn = self.feed_forward(x.transpose(1, 2))
        return self.norm3(x + self.dropout(ffn).transpose(1, 2)), weights_x, weights_memory


# class InformerDecoder(nn.Module):
#     def __init__(self, layers, norm_layer=None):
#         super().__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer

#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         for layer in self.layers:
#             x, _, _ = layer(x, cross, x_mask, cross_mask)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x


class InformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cross, x_mask, cross_mask=None):
        for layer in self.layers:
            x, _, _ = layer(x, cross, x_mask, cross_mask)

        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.cross_attn = MultiheadAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        attn, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        attn, _ = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn))
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, d_model, n_heads, d_ff, n_enc_layers, n_dec_layers, dropout=0.1, device=device):
        super().__init__()

        self.enc_embedding = None
        self.dec_embedding = None

        self.encoder = InformerEncoder(d_model, n_heads, d_ff, n_enc_layers)
        self.decoder = InformerDecoder(d_model, n_heads, d_ff, n_dec_layers)

        self.fc = nn.Linear(d_model, c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.fc(dec_out)

        return dec_out, attns



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        src_embedding = self.dropout(self.positional_encoding(self.encoder_embedding(src))).to(device)
        tgt_embedding = self.dropout(self.positional_encoding(self.encoder_embedding(tgt))).to(device)

        enc_output = src_embedding
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedding
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)


"""
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
n_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)

src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

epochs = 100
bar = pyprind.ProgBar(epochs, bar_char='█', monitor=True, stream=sys.stdout)
for epoch in range(epochs):
    optimizer.zero_grad()
    output = transformer(src_data.to(device), tgt_data[:, :-1].to(device))
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size).to(device), tgt_data[:, 1:].contiguous().view(-1).to(device))
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
"""