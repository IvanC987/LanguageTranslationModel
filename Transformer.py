import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, n_embd: int):
        super().__init__()
        pos_emb = torch.zeros((seq_len, n_embd), dtype=torch.float)  # (seq_len, n_embd)
        pos = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = 10_000 ** (torch.arange(0, n_embd, 2, dtype=torch.float) / n_embd)  # (n_embd//2)

        # Here, broadcasting is applied. By dividing shape (seq_len, 1) and (n_embd//2), we get (seq_len, n_embd//2)
        pos_emb[:, 0::2] = torch.sin(pos / div_term)  # For each even indices, we apply sin
        pos_emb[:, 1::2] = torch.cos(pos / div_term)  # For each odd, apply cos
        pos_emb = pos_emb.unsqueeze(0)  # Add batch dimension to ensure broadcasting. Current shape is (1, seq_len, n_embd)
        self.register_buffer("pos_emb", pos_emb)  # Add to buffer

    def forward(self, x: torch.tensor):
        # We look into the seq_len dimension and return the proper embeddings
        return self.pos_emb[:, :x.shape[1], :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, device: str):
        super().__init__()
        self.n_heads = n_heads
        assert n_embd % n_heads == 0, "n_embd % n_heads != 0"
        self.head_size = n_embd//n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.linear = nn.Linear(n_embd, n_embd)
        self.device = device

    def forward(self, x: torch.tensor, mask: bool):
        B, T, C = x.shape  # Batch are always the same, with the exception when inferencing/translating, which is B=1
        qkv_layers = self.qkv(x)  # (batch, seq_len, 3 * n_embd)
        qkv_layers = qkv_layers.reshape(B, T, self.n_heads, 3 * self.head_size)
        qkv_layers = qkv_layers.permute(0, 2, 1, 3)  # (batch, n_heads, seq, head_size * 3)
        q, k, v = qkv_layers.chunk(3, dim=-1)  # It is now (batch, n_heads, seq, head_size) for each q, k, v

        # Now computing the self attention
        y = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # Mat mul by last 2 dims. (seq, head_size) @ (head_size, seq) = (seq, seq)
        if mask:  # Apply causal mask when in Decoder Self Attention
            m = torch.tril(torch.ones((T, T), dtype=torch.bool, device=self.device))
            y = y.masked_fill(~m, -1e9)
        y = F.softmax(y, dim=-1)
        values = y @ v  # The result should be of original shape, (batch, n_heads, seq, head_size)

        values = values.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_size)  # Return back to original shape
        out = self.linear(values)  # Pass through a final linear layer
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        assert n_embd % n_heads == 0, "n_embd % n_heads != 0"
        self.head_size = n_embd//n_heads
        self.kv = nn.Linear(n_embd, 2 * n_embd)
        self.q = nn.Linear(n_embd, n_embd)
        self.linear = nn.Linear(n_embd, n_embd)

    def forward(self, enc: torch.tensor, dec: torch.tensor):
        # Essentially the same as MultiHeadSelfAttention class, only difference is k, v comes from Encoder and no masking needed
        B, T, C = enc.shape  # enc and dec should be of same shape
        kv = self.kv(enc)
        q = self.q(dec)
        kv = kv.reshape(B, T, self.n_heads, 2 * self.head_size)
        q = q.reshape(B, T, self.n_heads, self.head_size)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        q = q.permute(0, 2, 1, 3)

        # Apply Attention Mechanism
        y = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        y = F.softmax(y, dim=-1)
        values = y @ v

        # Result should be shape (B, T, n_head * head_size) after permute and reshape
        values = values.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_size)
        out = self.linear(values)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # A simple feedforward layer based on the original paper
        self.seq = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.tensor):
        return self.seq(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float, device: str):
        super().__init__()
        # Here we define a single Encoder Layer
        self.self_mh = MultiHeadSelfAttention(n_embd=n_embd, n_heads=n_heads, device=device)
        self.norm1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.tensor):
        x = x + self.dropout1(self.self_mh(self.norm1(x), mask=False))
        x = x + self.dropout2(self.ffwd(self.norm2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, n_layers: int, dropout: float, device: str):
        super().__init__()
        # Here's the entire Encoder Block, comprised of n_layers
        self.enc_layers = nn.Sequential(*[EncoderLayer(n_embd=n_embd, n_heads=n_heads, dropout=dropout, device=device) for _ in range(n_layers)])

    def forward(self, x: torch.tensor):
        # Pass it through sequentially
        x = self.enc_layers(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float, device: str):
        super().__init__()
        # Defining a single Decoder Layer
        self.self_mh = MultiHeadSelfAttention(n_embd=n_embd, n_heads=n_heads, device=device)
        self.norm1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(p=dropout)

        self.cross_mh = MultiHeadCrossAttention(n_embd=n_embd, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(p=dropout)

        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
        self.norm3 = nn.LayerNorm(n_embd)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, enc_output: torch.tensor, x: torch.tensor):
        x = x + self.dropout1(self.self_mh(self.norm1(x), mask=True))
        x = x + self.dropout2(self.cross_mh(self.norm2(enc_output), self.norm2(x)))
        x = x + self.dropout3(self.ffwd(self.norm3(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, n_layers: int, dropout: float, device: str):
        super().__init__()
        # Defining the entire Decoder Block
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(n_embd=n_embd, n_heads=n_heads, dropout=dropout, device=device) for _ in range(n_layers)]
        )

    def forward(self, enc_output: torch.tensor, x: torch.tensor):
        # In the forward pass, we pass the original enc_output along with each updated dec_output, which is "x"
        for i in range(len(self.dec_layers)):
            x = self.dec_layers[i](enc_output, x)
        return x


class SentenceEmbedding(nn.Module):
    def __init__(self, seq_len: int, n_embd: int, language_to_int: dict, start_token: str, pad_token: str, end_token: str, device: str):
        super().__init__()
        self.n_embd = n_embd
        self.emb = nn.Embedding(len(language_to_int), n_embd)  # (vocab_size, n_embd), len(language_to_int) would be the vocab_size
        self.pos_enc = PositionalEncoding(seq_len, n_embd)  # (seq_len, n_embd)
        self.seq_len = seq_len
        self.language_to_int = language_to_int
        self.start_token = start_token
        self.pad_token = pad_token
        self.end_token = end_token
        self.device = device

    def forward(self, x: list[str], add_start: bool, add_end: bool):
        # Given x, which should be a list of strings, we will need to tokenize them and apply self.emb + self.pos_enc
        tokens = self.batch_tokenize(x, add_start=add_start, add_end=add_end)
        tok_emb = self.emb(tokens) * (self.n_embd ** 0.5)
        pos = self.pos_enc(tokens)
        return tok_emb + pos

    def batch_tokenize(self, batch_sentences: list[str], add_start: bool, add_end: bool):
        # Expects a list (batch) of sentences along with if needed to add a start/end token to each sentence
        def tokenize(sentence: str, add_start: bool, add_end: bool):
            result = [self.language_to_int[c] for c in sentence]
            if add_start:
                result.insert(0, self.language_to_int[self.start_token])
            if add_end:
                result.append(self.language_to_int[self.end_token])
            while len(result) < self.seq_len:
                result.append(self.language_to_int[self.pad_token])

            assert len(result) == self.seq_len
            return result

        return torch.tensor([tokenize(s, add_start, add_end) for s in batch_sentences], dtype=torch.long, device=self.device)


class Transformer(nn.Module):
    # Define the overall Transformer class
    def __init__(self, seq_len: int, n_embd: int, n_heads: int, n_layers: int, dropout: float, device: str,
                 str_to_int: dict, int_to_str: dict, start_token: str, pad_token: str, end_token: str):
        super().__init__()
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.device = device
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.start_token = start_token
        self.pad_token = pad_token
        self.end_token = end_token

        self.src_embd = SentenceEmbedding(seq_len=seq_len, n_embd=n_embd, language_to_int=str_to_int, start_token=start_token, pad_token=pad_token, end_token=end_token, device=device)
        self.tgt_embd = SentenceEmbedding(seq_len=seq_len, n_embd=n_embd, language_to_int=str_to_int, start_token=start_token, pad_token=pad_token, end_token=end_token, device=device)
        self.enc_block = EncoderBlock(n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, dropout=dropout, device=device)
        self.dec_block = DecoderBlock(n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, dropout=dropout, device=device)
        self.projection = nn.Linear(n_embd, len(str_to_int))

        # As per the paper, src embedding, tgt embedding, and linear projection should share the same weights
        self.src_embd.emb.weight = self.tgt_embd.emb.weight
        self.projection.weight = self.tgt_embd.emb.weight

    def forward(self, enc_input: list[str], dec_input: list[str], add_enc_start: bool, add_enc_end: bool, add_dec_start: bool, add_dec_end: bool):
        enc_x = self.src_embd(enc_input, add_enc_start, add_enc_end)
        enc_output = self.enc_block(enc_x)

        dec_x = self.tgt_embd(dec_input, add_dec_start, add_dec_end)
        dec_output = self.dec_block(enc_output, dec_x)

        output = self.projection(dec_output)
        return output

    def translate(self, src: str, k=10) -> str:
        tgt = ""  # Start out with empty string

        for char in src:  # Checking for existence of all characters in provided src sentence
            if char not in self.str_to_int:
                print(f"Value \"{char}\" not in src_vocab!")
                return ""

        for i in range(self.seq_len):  # Iterate until we get EOS token or at seq_len
            logits = self([src], [tgt], add_enc_start=False, add_enc_end=False, add_dec_start=True, add_dec_end=False)
            logits = logits[0]  # (First batch)

            top_k_logits, top_k_idx = torch.topk(logits, k=k)  # Using top-k method for sampling tokens
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            top_k_idx = top_k_idx[i]
            current_token = top_k_probs[i]

            token = torch.multinomial(current_token, num_samples=1)
            token_index = top_k_idx[token].item()  # Get the original index of the sampled token
            char = self.int_to_str[token_index]

            # Ensure not to select special tokens
            while char == self.start_token or char == self.pad_token:
                token = torch.multinomial(current_token, num_samples=1)
                token_index = top_k_idx[token].item()
                char = self.int_to_str[token_index]

            if char == self.end_token:  # If we get EOS token, we return
                return tgt
            tgt = tgt + char

        return tgt  # If we reach this point, meaning len(tgt) == seq_len, return



