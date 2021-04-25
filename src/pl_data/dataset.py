import itertools

from omegaconf import DictConfig, ValueNode
import pandas as pd
from spacy.training import iob_to_biluo
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from allennlp_models.pretrained import load_predictor
from src.common.utils import Config, convert_examples_to_features, encode_labels


class CoNLLDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        cfg: DictConfig,
        tokenizer=AutoTokenizer,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.name = name

        self.data: list[dict[str, tuple[str]]] = self.read_conll()
        self.tokenizer = tokenizer.from_pretrained(
            Config.MODEL_NAME, add_prefix_space=True
        )
        self.max_token_len = Config.MAX_TOKEN_LEN

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        tokens: list[str] = list(self.data[index]["tokens"])
        labels_id: list[int] = [
            Config.GER_LABELS[label] for label in self.data[index]["tags"]
        ]
        encoding, labels_encoded = encode_labels(tokens, labels_id, self.tokenizer)
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.LongTensor(labels_encoded),
        )

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"

    def read_conll(self) -> list[dict[str, tuple[str]]]:
        with open(self.path, "r") as conll_file:  # type: ignore
            data: list[dict[str, tuple[str]]] = []
            for divider, lines in itertools.groupby(
                conll_file, lambda x: x.startswith("-DOCSTART-")
            ):
                if divider:
                    continue
                fields = [line.strip().split() for line in lines]
                fields = [line for line in fields if line]
                fields = [line for line in zip(*fields)]
                tokens, _, _, tags = fields
                sequence = {"tokens": tokens, "tags": iob_to_biluo(tags)}
                data.append(sequence)
        return data



class TextDataset(Dataset):
    def __init__(
        self,
        text: list,
        coref_model: str,
        tokenizer=AutoTokenizer,
    ):
        predictor = load_predictor(coref_model)
        predictor.cuda_device = 0 if torch.cuda.is_available else -1

        self.text: list[str] = text
        self.text = [predictor.coref_resolved(i) for i in tqdm(text)]

        self.tokenizer = tokenizer.from_pretrained(
            Config.MODEL_NAME, add_prefix_space=True
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index: int):
        text: str = self.text[index]
        encoding = self.tokenizer(
            text,
            return_attention_mask=True,
            max_length=Config.MAX_TOKEN_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
        )


class CSVDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        cfg: DictConfig,
        tokenizer=AutoTokenizer,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.data: pd.DataFrame = pd.read_csv(
            path, header=None, names=["label", "text_a"]  # type: ignore
        )

        ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer = tokenizer.from_pretrained(Config.MODEL_NAME)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data.iloc[index]

        item.label = Config.REL_LABELS[item.label]
        features = convert_examples_to_features(
            item, max_seq_len=Config.MAX_TOKEN_LEN, tokenizer=self.tokenizer
        )
        return {
            "input_ids": torch.tensor(features.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(features.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(features.token_type_ids, dtype=torch.long),
            "labels": torch.tensor(features.label_id, dtype=torch.long),
            "e1_mask": torch.tensor(features.e1_mask, dtype=torch.long),
            "e2_mask": torch.tensor(features.e2_mask, dtype=torch.long),
        }
