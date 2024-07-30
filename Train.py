import torch
from torch import nn
import random
from Transformer import Transformer
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device}")


# -------------------------------------------------
# Defining Hyperparameters
batch_size = 32
seq_len = 192
n_embd = 256
n_heads = 8
n_layers = 4
dropout = 0.1

training_iterations = 5_000
warmup_iterations = min(4_000, int(training_iterations*0.1))
eval_interval = 500
eval_iterations = 10
# -------------------------------------------------


src_filepath = "en_dataset.txt"
tgt_filepath = "zh_dataset.txt"

SOS_TOKEN = "<SOS>"
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"


# Loading Data
def read_text_file():
    # This function reads in two text files, assuming each example is separated by a newline
    with open(src_filepath, "r", encoding="utf-8") as f:
        src_text = f.read()
        src_sentences = src_text.split("\n")

    with open(tgt_filepath, "r", encoding="utf-8") as f:
        tgt_text = f.read()
        tgt_sentences = tgt_text.split("\n")

    assert len(src_sentences) >= 1000, "Insufficient data. Len(dataset) < 1000!"
    assert len(src_sentences) == len(tgt_sentences), "Number of src sentences != tgt sentences!"

    unique_chars = [SOS_TOKEN] + sorted(list(set(src_text + tgt_text))) + [PAD_TOKEN, EOS_TOKEN]  # Add the special tokens

    str_to_int = {c: i for i, c in enumerate(unique_chars)}  # Converting from character to integer
    int_to_str = {i: c for i, c in enumerate(unique_chars)}  # Converting from integer to character

    return str_to_int, int_to_str, src_sentences, tgt_sentences


def filter_max_length(src_sentences: list[str], tgt_sentences: list[str]):
    # Filters and truncates sentences with length longer than seq_len-2  (Assuming we need SOS and EOS tokens)

    assert len(src_sentences) == len(tgt_sentences), "len(src_sentences) != len(tgt_sentences)"

    src, tgt = [], []
    for idx in range(len(src_sentences)):
        if (len(src_sentences[idx]) < (seq_len - 1)) and (len(tgt_sentences[idx]) < (seq_len - 1)):
            src.append(src_sentences[idx])
            tgt.append(tgt_sentences[idx])
        else:
            src.append(src_sentences[idx][:seq_len-2])
            tgt.append(tgt_sentences[idx][:seq_len-2])

    return src, tgt


def get_batch(x_data: list[str], y_data: list[str], size: int = batch_size):
    # Get a random batch of training examples
    assert len(x_data) == len(y_data), "len(x_data) != len(y_data)"
    idx = torch.randint(len(x_data), (size,))
    x, y = [], []
    for i in idx:
        x.append(x_data[i])
        y.append(y_data[i])
    return x, y


@torch.no_grad()
def estimate_loss():
    # Returns a dictionary containing estimated training and validation losses
    out = {}
    model.eval()
    for split in ["train", "val"]:
        all_losses = torch.zeros(eval_iterations)

        for i in range(eval_iterations):
            x, y = get_batch(src_train, tgt_train) if split == "train" else get_batch(src_val, tgt_val)
            logits = model(x, y, add_enc_start=False, add_enc_end=False, add_dec_start=True, add_dec_end=True)
            labels = model.tgt_embd.batch_tokenize(batch_sentences=y, add_start=False, add_end=True).to(device)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), labels.view(B * T))
            all_losses[i] = loss.item()

        out[split] = all_losses.mean()
    model.train()
    return out


def evaluate_model(iteration: int, x: list[str], y: list[str], logits: torch.tensor):
    out = estimate_loss()
    print("-------------------------------------")
    print(f"At iteration {iteration}, Training Loss = {round(float(out["train"]), 4)}, Validation Loss = {round(float(out["val"]), 4)}")
    print(f"Src Sentence: {x[0]}")
    print(f"Tgt Sentence: {y[0]}")

    # Generating prediction based on above sentence
    sentence = logits[0].squeeze(dim=0)  # Get the corresponding first batch
    sentence = torch.softmax(sentence, dim=-1)  # Softmax to convert to probabilities
    all_idx = []
    for j in range(sentence.shape[0]):
        current_idx = torch.multinomial(sentence[j], num_samples=1)
        if current_idx == str_to_int[EOS_TOKEN]:
            break
        all_idx.append(current_idx.item())
    print(f"Prediction Sentence: {"".join([int_to_str[k] for k in all_idx])}")

    print("**********************************")

    # Testing out the current progress
    examples = ["What is your name?", "He went to school.", "This is my friend Chris.", "Where are you going?", "How is your day today?", "Nice to meet you. My name is John.", "Are you sure?!?", "She is going to the US"]
    txt = random.choice(examples)
    translation = model.translate(txt)
    print(f"Testing")
    print(f"Src: {txt}")
    print(f"Translation: {translation}")
    print("-------------------------------------")

    return float(out["train"]), float(out["val"])


def adjust_lr(step: int):
    # Adjusting the learning rate as described in the original paper, "Attention Is All You Need"
    if step == 0:
        lr = 1e-4
    else:
        lr = (n_embd ** -0.5) * min(step**-0.5, step * (warmup_iterations**-1.5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def initiate_model():
    str_to_int, int_to_str, src_sentences, tgt_sentences = read_text_file()
    src_sentences, tgt_sentences = filter_max_length(src_sentences=src_sentences, tgt_sentences=tgt_sentences)

    # Splitting data into train and val datasets
    n = int(len(src_sentences) * 0.9)
    src_train, src_val, tgt_train, tgt_val = src_sentences[:n], src_sentences[n:], tgt_sentences[:n], tgt_sentences[n:]

    model = Transformer(seq_len=seq_len, n_embd=n_embd, n_heads=n_heads, n_layers=n_layers,
                        dropout=dropout, device=device, str_to_int=str_to_int, int_to_str=int_to_str,
                        start_token=SOS_TOKEN, pad_token=PAD_TOKEN, end_token=EOS_TOKEN).to(device)

    # Simple weight initialization using the Xavier init
    for params in model.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

    model.train()
    return model, str_to_int, int_to_str, src_train, src_val, tgt_train, tgt_val


# ***********************
# NOW IN THE TRAINING PHASE
# ***********************


if __name__ == "__main__":
    model, str_to_int, int_to_str, src_train, src_val, tgt_train, tgt_val = initiate_model()

    # Print the number of parameters in the model
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    load_model = input("Load a saved model to continue training? (Y/N): ")
    while load_model.lower() != "y" and load_model.lower() != "n":
        print("Invalid response. Enter Y/N")
        load_model = input("Load a saved model to continue training? (Y/N): ")
    if load_model == 'y':
        state_path = input("Enter the path to the saved model: ")
        print("Loading model...")
        state = torch.load(state_path, map_location=torch.device(device))
        model.load_state_dict(state)
    else:
        print("Creating new model...")


    criterion = nn.CrossEntropyLoss(ignore_index=str_to_int[PAD_TOKEN])
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    model.train()


    all_losses = []  # Used for visual plotting at the end of training
    saved_batches = []  # Used to located possible bad src/translation sentences, identified by all_losses graph
    train_loss, val_loss = 0, 0
    for i in range(training_iterations):
        x, y = get_batch(x_data=src_train, y_data=tgt_train)
        logits = model(x, y, add_enc_start=False, add_enc_end=False, add_dec_start=True, add_dec_end=True)
        labels = model.tgt_embd.batch_tokenize(batch_sentences=y, add_start=False, add_end=True).to(device)

        B, T, C = logits.shape
        loss = criterion(logits.view(B*T, C), labels.view(B*T))

        adjust_lr(step=i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % eval_interval == 0:
            train_loss, val_loss = evaluate_model(iteration=i, x=x, y=y, logits=logits)

            # Might want to intercede if difference between training loss and val loss are past a certain threshold
            if abs(train_loss - val_loss) > 0.2 * train_loss:
                stop = input("Exit training? (Y/N): ")
                while stop.lower() != "y" and stop.lower() != "n":
                    print("Invalid response. Enter Y/N")
                    stop = input("Exit training? (Y/N): ")
                if stop.lower() == "y":
                    break

        all_losses.append(loss.item())
        saved_batches.append((x, y))


    plt.plot(all_losses, c="blue")
    plt.show()


    save_model = input("Save Model? (Y/N): ")
    while save_model.lower() != "y" and save_model.lower() != "n":
        print("Invalid response. Enter Y/N")
        save_model = input("Save Model? (Y/N): ")
    if save_model.lower() == "y":
        name = f"TL_{train_loss:.3f}-VL_{val_loss:.3f}_state_dict"
        torch.save(model.state_dict(), name + ".pth")


    with torch.no_grad():
        print("Enter \"q\" to exit")
        model.eval()
        while True:
            src_text = input("Enter src text: ")

            if src_text.lower() == "q":
                break

            translation = model.translate(src_text)
            print(f"Translation: {translation}")
            print("---------------------------------------")

