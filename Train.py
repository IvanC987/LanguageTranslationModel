import torch
from torch import nn
from torch.nn import functional as F
from Transformer import Transformer


torch.manual_seed(89)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device}")

# -------------------------------------------------
# Defining Hyperparameters
# Currently setting all values to be minimal for faster testing
batch_size = 4
seq_len = 50
n_embd = 64
n_heads = 4
n_layers = 1
training_iterations = 2001
eval_interval = 100
lr = 1e-4

# -------------------------------------------------

SOS_TOKEN = "<SOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"


# Tokenization function
def char_level_tokenizer(src_text, tgt_text, src_list, tgt_list):
    unique_src_char = sorted(list(set(src_text)))
    unique_src_char = [SOS_TOKEN] + unique_src_char + [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
    unique_tgt_char = sorted(list(set(tgt_text)))
    unique_tgt_char = [SOS_TOKEN] + unique_tgt_char + [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]

    src_vocab_size = len(unique_src_char)
    tgt_vocab_size = len(unique_tgt_char)

    src_to_int = {c: i for i, c in enumerate(unique_src_char)}
    int_to_src = {i: c for i, c in enumerate(unique_src_char)}
    tgt_to_int = {c: i for i, c in enumerate(unique_tgt_char)}
    int_to_tgt = {i: c for i, c in enumerate(unique_tgt_char)}

    # Here, I am tokenizing each sentence, truncating if longer than seq_len, padding if shorter than seq_len
    src_tokens, tgt_tokens = [], []
    for src_sentence in src_list:
        temp = [src_to_int[PAD_TOKEN] for _ in range(max(seq_len, len(src_sentence) + 1))]  # This method is better if the majority of sentence length < seq_len
        temp[0] = src_to_int[SOS_TOKEN]
        for c in range(len(src_sentence)):
            temp[c+1] = src_to_int[src_sentence[c]]
        temp = temp[:seq_len]
        temp[min(seq_len-1, len(src_sentence)+1)] = src_to_int[EOS_TOKEN]
        src_tokens.append(temp)

    for tgt_sentence in tgt_list:
        temp = [tgt_to_int[PAD_TOKEN] for _ in range(max(seq_len, len(tgt_sentence) + 1))]  # This method is better if the majority of sentence length < seq_len
        temp[0] = tgt_to_int[SOS_TOKEN]
        for c in range(len(tgt_sentence)):
            temp[c+1] = tgt_to_int[tgt_sentence[c]]
        temp = temp[:seq_len]
        temp[min(seq_len-1, len(tgt_sentence)+1)] = tgt_to_int[EOS_TOKEN]
        tgt_tokens.append(temp)

    return src_vocab_size, tgt_vocab_size, src_to_int, int_to_src, tgt_to_int, int_to_tgt, src_tokens, tgt_tokens


# Loading Data
def load_data(src_filepath, tgt_filepath):
    with open(src_filepath, "r", encoding="utf-8") as f:
        src_data = f.read()
        src_sentences = src_data.split("\n")

    with open(tgt_filepath, "r", encoding="utf-8") as f:
        tgt_data = f.read()
        tgt_sentences = tgt_data.split("\n")

    assert len(src_sentences) >= 1000, "Insufficient data. Len(dataset) < 1000!"
    assert len(src_sentences) == len(tgt_sentences), "Number of src sentences != tgt sentences!"

    src_vocab_size, tgt_vocab_size, src_to_int, int_to_src, tgt_to_int, int_to_tgt, src_tokens, tgt_tokens = char_level_tokenizer(
        src_text=src_data, tgt_text=tgt_data, src_list=src_sentences, tgt_list=tgt_sentences)

    src_tokens = torch.tensor(src_tokens)
    tgt_tokens = torch.tensor(tgt_tokens)

    n = int(len(src_tokens) * 0.9) if len(src_tokens) >= 10_000 else int(len(src_tokens) * 0.8)
    src_train, src_val, tgt_train, tgt_val = src_tokens[:n], src_tokens[n:], tgt_tokens[:n], tgt_tokens[n:]

    return src_vocab_size, src_train, src_val, src_to_int, int_to_src, tgt_vocab_size, tgt_train, tgt_val, tgt_to_int, int_to_tgt


def decode(given_list, decoder):
    result = "".join([decoder[i] for i in given_list])
    for special_token in [SOS_TOKEN, PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]:
        result = result.replace(special_token, "")
    return result


def get_batch(x_data, y_data, size=batch_size):
    assert len(x_data) == len(y_data), "len(x_data) != len(y_data)"
    idx = torch.randint(len(x_data), (size,))
    x, y = x_data[idx].to(device), y_data[idx].to(device)
    return x, y


# ***********************
# NOW IN THE TRAINING PHASE
# ***********************

src_vocab_size, src_train, src_val, src_to_int, int_to_src, tgt_vocab_size, tgt_train, tgt_val, tgt_to_int, int_to_tgt = load_data(
    src_filepath="en_adjusted.txt", tgt_filepath="zh_adjusted.txt")


model = Transformer(src_vocab_size, tgt_vocab_size, batch_size, seq_len, n_embd, n_heads, n_layers, device)
model = model.to(device)
model.train()


criterion = nn.CrossEntropyLoss(ignore_index=tgt_to_int[PAD_TOKEN], reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for i in range(training_iterations):
    optimizer.zero_grad()
    x, y = get_batch(x_data=src_train, y_data=tgt_train)
    logits = model(x, y)

    B, T, C = logits.shape
    loss = criterion(logits.view(B*T, C), y.view(B*T))
    valid_tokens = torch.where(y.view(-1) == tgt_to_int[PAD_TOKEN], False, True)
    loss = loss.sum() / valid_tokens.sum()

    loss.backward()
    optimizer.step()

    if i % eval_interval == 0:
        print("\n-------------------------------------")
        print(f"At iteration {i}, Training Loss = {loss.item()}")
        print(f"English: {decode(x[0].tolist(), int_to_src)}")
        print(f"Target: {decode(y[0].tolist(), int_to_tgt)}")

        probs = F.softmax(logits[0], dim=-1)
        result = []
        for i in range(len(probs)):
            result.append(torch.multinomial(probs[i], num_samples=1).item())
        print(f"Prediction: {decode(result, int_to_tgt)}")
        print("-------------------------------------\n")
