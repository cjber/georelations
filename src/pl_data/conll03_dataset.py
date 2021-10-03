import torch
from datasets import load_dataset
from datasets.arrow_dataset import concatenate_datasets
from pathlib import Path
from src.common.utils import Const, encode_labels
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


def use_loc(example):
    new_tags = []
    for tag in example["ner_tags"]:
        if tag == 5:
            tag = 1
        elif tag == 6:
            tag = 2
        else:
            tag = 0
        new_tags.append(tag)
    example["ner_tags"] = new_tags
    return example


class CoNLL03Dataset(Dataset):
    def __init__(self, path: Path, tokenizer=AutoTokenizer):
        super().__init__()
        self.model_name = Const.MODEL_NAME
        self.max_token_len = Const.MAX_TOKEN_LEN

        conll = load_dataset("conll2003", data_dir=str(path))
        conll = concatenate_datasets(
            [conll["train"], conll["validation"], conll["test"]]
        )
        self.data = conll.map(use_loc)
        self.data = self.data.filter(lambda example: 1 in example["ner_tags"])
        self.tokenizer = tokenizer.from_pretrained(
            self.model_name, add_prefix_space=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        idx = self.data[index]
        tokens = idx["tokens"]
        tags = idx["ner_tags"]

        encoding, labels_encoded = encode_labels(
            tokens, tags, self.tokenizer, self.max_token_len
        )
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.LongTensor(labels_encoded).flatten(),
        )
