import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, n_embd):
        super().__init__()
        pos_emb = torch.zeros((seq_len, n_embd), dtype=torch.float)
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = 10_000 ** (torch.arange(0, n_embd, 2, dtype=torch.float) / n_embd)

        pos_emb[:, 0::2] = torch.sin(pos / div_term)
        pos_emb[:, 1::2] = torch.cos(pos / div_term)
        pos_emb = pos_emb.unsqueeze(0)  # To ensure broadcasting
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x):
        return self.pos_emb[:, :x.shape[1], :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, batch_size, seq_len, n_embd, n_heads, device):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.n_heads = n_heads
        assert n_embd % n_heads == 0, "n_embd % n_heads != 0"
        self.head_size = n_embd//n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.linear = nn.Linear(n_embd, n_embd)
        self.device = device

    def forward(self, x, mask):
        qkv_layers = self.qkv(x)
        qkv_layers = qkv_layers.reshape(self.batch_size, self.seq_len, self.n_heads, 3 * self.head_size)
        qkv_layers = qkv_layers.permute(0, 2, 1, 3)  # It will now be (batch, n_heads, seq, head_size * 3)
        q, k, v = qkv_layers.chunk(3, dim=-1)  # It will now be (batch, n_heads, seq, head_size) each

        # Now computing the self attention
        y = q @ k.transpose(-2, -1) / (self.n_embd ** 0.5)  # Mat mul by last 2 dims. (seq, head_size) @ (head_size, seq) = (seq, seq)
        if mask:
            m = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool, device=self.device))
            y = y.masked_fill(~m, -1e9)
        y = F.softmax(y, dim=-1)
        values = y @ v  # The result should be of same shape, (batch, n_heads, seq, head_size)

        values = values.permute(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.n_heads * self.head_size)
        out = self.linear(values)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, batch_size, seq_len, n_embd, n_heads):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.n_heads = n_heads
        assert n_embd % n_heads == 0, "n_embd % n_heads != 0"
        self.head_size = n_embd//n_heads
        self.kv = nn.Linear(n_embd, 2 * n_embd)
        self.q = nn.Linear(n_embd, n_embd)
        self.linear = nn.Linear(n_embd, n_embd)

    def forward(self, enc, dec):
        kv = self.kv(enc)
        q = self.q(dec)
        kv = kv.reshape(self.batch_size, self.seq_len, self.n_heads, 2 * self.head_size)
        q = q.reshape(self.batch_size, self.seq_len, self.n_heads, self.head_size)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        q = q.permute(0, 2, 1, 3)

        # Now computing the self attention
        y = q @ k.transpose(-2, -1) / (self.n_embd ** 0.5)  # Mat mul by last 2 dims. (seq, head_size) @ (head_size, seq) = (seq, seq)
        y = F.softmax(y, dim=-1)
        values = y @ v  # The result should be of same shape, (batch, n_heads, seq, head_size)

        # Do the math. Result should be shape (eB, dT, n_head * head_size) after permute and reshape
        values = values.permute(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.n_heads * self.head_size)
        out = self.linear(values)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        return self.seq(x)


class EncoderLayer(nn.Module):
    def __init__(self, batch_size, seq_len, n_embd, n_heads, device):
        super().__init__()
        self.self_mh = MultiHeadSelfAttention(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, n_heads=n_heads, device=device)
        self.norm1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(p=0.1)

        self.ffwd = FeedForward(n_embd=n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x + self.dropout1(self.self_mh(self.norm1(x), mask=False))
        x = x + self.dropout2(self.ffwd(self.norm2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, batch_size, seq_len, n_embd, n_heads, n_layers, device):
        super().__init__()
        self.enc_layers = nn.Sequential(*[EncoderLayer(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, n_heads=n_heads, device=device) for _ in range(n_layers)])

    def forward(self, x):
        x = self.enc_layers(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, batch_size, seq_len, n_embd, n_heads, device):
        super().__init__()
        self.self_mh = MultiHeadSelfAttention(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, n_heads=n_heads, device=device)
        self.norm1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(p=0.1)  # CHANGE LATER AS PARAMETER!

        self.cross_mh = MultiHeadCrossAttention(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(p=0.1)

        self.ffwd = FeedForward(n_embd=n_embd)
        self.norm3 = nn.LayerNorm(n_embd)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, enc_output, x):
        x = x + self.dropout1(self.self_mh(self.norm1(x), mask=True))
        x = x + self.dropout2(self.cross_mh(self.norm2(enc_output), self.norm2(x)))
        x = x + self.dropout3(self.ffwd(self.norm3(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, batch_size, seq_len, n_embd, n_heads, n_layers, device):
        super().__init__()
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, n_heads=n_heads, device=device) for _ in range(n_layers)]
        )

    def forward(self, enc_output, x):
        for i in range(len(self.dec_layers)):
            x = self.dec_layers[i](enc_output, x)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, batch_size, seq_len, n_embd, n_heads, n_layers, device):
        super().__init__()
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.device = device
        self.src_embd = nn.Embedding(src_vocab_size, n_embd)
        self.tgt_embd = nn.Embedding(tgt_vocab_size, n_embd)
        self.pos_enc = PositionalEncoding(seq_len=seq_len, n_embd=n_embd)
        self.enc_block = EncoderBlock(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, device=device)
        self.dec_block = DecoderBlock(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, device=device)
        self.projection = nn.Linear(n_embd, tgt_vocab_size)

    def forward(self, enc_tokens, dec_tokens):
        enc_embd = self.src_embd(enc_tokens) * (self.n_embd ** 0.5)
        enc_pe = self.pos_enc(enc_tokens)
        enc_x = enc_embd + enc_pe
        enc_output = self.enc_block(enc_x)

        dec_embd = self.tgt_embd(dec_tokens)
        dec_pe = self.pos_enc(dec_tokens)
        dec_x = dec_embd + dec_pe
        dec_output = self.dec_block(enc_output, dec_x)

        output = self.projection(dec_output)
        return output

    def generate(self, enc_text, dec_text, max_new_tokens, src_to_int, tgt_to_int):
        # Need to add padding tokens?
        enc_tokens = torch.tensor([src_to_int[c] for c in enc_text][:self.seq_len], device=self.device).reshape(1, -1)  # Limited to seq_len due to pos_enc limitation
        # dec_tokens = torch.tensor([tgt_to_int[c] for c in dec_text][:self.seq_len]).reshape(1, -1)
        dec_tokens = torch.ones((1, 1), dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            trim = dec_tokens[:, -self.seq_len:]
            logits = self(enc_tokens, trim)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            dec_tokens = torch.cat((dec_tokens, idx), dim=-1)

        return dec_tokens


