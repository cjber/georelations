from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl
import nltk

dataset = read_jsonl(
    filepath="labelling/project_1_dataset.jsonl", dataset=NERDataset, encoding="utf-8"
)

conll_dataset = dataset.to_conll2003(tokenizer=nltk.word_tokenize)

try:
    with open("labelling/ner_dataset.conll", "w") as f:
        for item in conll_dataset:
            f.write(item["data"])
except LookupError:
    nltk.download("punkt")
    with open("labelling/dataset.conll", "w") as f:
        for item in conll_dataset:
            f.write(item["data"])
