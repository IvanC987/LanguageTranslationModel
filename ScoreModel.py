import jieba
import torch
from nltk.translate.meteor_score import meteor_score

with open("en_test.txt", "r", encoding="utf-8") as f:
    src_sentences = f.read().split("\n")

with open("zh_test.txt", "r", encoding="utf-8") as f:
    tgt_sentences = f.read().split("\n")


assert len(src_sentences) == len(tgt_sentences), f"Len src: {len(src_sentences)}, Len tgt: {len(tgt_sentences)}"
print(f"Length of testing dataset: {len(src_sentences)}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("TL_0.810-VL_0.955.pth").to(device)

print("Starting")

with torch.no_grad():
    model.eval()
    hypotheses = []
    references = []
    c = 0
    with torch.no_grad():
        for source, target in zip(src_sentences, tgt_sentences):
            translated = model.translate(source)
            hypotheses.append(translated)
            references.append(target.split("_")[:-1])

            c += 1
            if c % 100 == 0:
                print(f"At sentence {c}")


# Unlike western languages, Chinese doesn't use spaces between characters, here the sentences are tokenized using jieba
tokenized_hypotheses = [list(jieba.cut(hyp)) for hyp in hypotheses]
tokenized_references = [[list(jieba.cut(ref)) for ref in refs] for refs in references]

# Calculate METEOR score
scores = [meteor_score(refs, hyp) for refs, hyp in zip(tokenized_references, tokenized_hypotheses)]
average_score = sum(scores) / len(scores)
print(f'Average METEOR score: {average_score:.4f}')

