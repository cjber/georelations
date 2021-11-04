import torch
from datasets.arrow_dataset import concatenate_datasets
from datasets.load import load_dataset
from ekphrasis.classes.preprocessor import TextPreProcessor
from pathlib import Path
from spacy.training.iob_utils import iob_to_biluo
from src.common.utils import Const, Label, encode_labels
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Union


class WNUTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: Union[Path, str],
        tokenizer: PreTrainedTokenizer = AutoTokenizer,  # type: ignore
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer.from_pretrained(
            Const.MODEL_NAME, add_prefix_space=True
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )
        self.text_processor = TextPreProcessor(**Const.TEXT_PROCESSOR_ARGS)

        if path == "train":
            wnut = load_dataset(
                "wnut_17", split={"train": "train", "validation": "validation"}  # type: ignore
            )
            wnut = concatenate_datasets([wnut["train"], wnut["validation"]])  # type: ignore
        elif path == "test":
            wnut = load_dataset("wnut_17", split="test")
        else:
            raise ValueError
        wnut = wnut.map(self.use_loc).map(self.normalise).map(self.to_biluo)  # type: ignore

        locs = wnut.filter(
            lambda example: any(x in [1, 4] for x in example["ner_tags"])
        )

        # reduce imbalance by keeping 1/8 of sentences w/o locations
        all_other = wnut.filter(
            lambda example: all(x not in [1, 4] for x in example["ner_tags"])
        ).shard(  # type: ignore
            8, 0
        )
        self.data = concatenate_datasets([locs, all_other])  # type: ignore

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx: dict[str, list] = self.data[index]

        encoding, labels_encoded = encode_labels(
            idx["tokens"],
            idx["ner_tags"],
            self.tokenizer,
            Const.MAX_TOKEN_LEN,
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels_encoded,
        }

    @staticmethod
    def use_loc(example: dict[str, list[int]]) -> dict[str, list[int]]:
        new_tags = []
        for tag in example["ner_tags"]:
            if tag == 7:
                new_tag = 1
            elif tag == 8:
                new_tag = 2
            else:
                new_tag = 0
            new_tags.append(new_tag)
        example["ner_tags"] = new_tags
        return example

    def normalise(self, example: dict[str, list[str]]) -> dict[str, list[str]]:
        example["tokens"] = [
            "".join(self.text_processor.pre_process_doc(word))
            for word in example["tokens"]
        ]
        return example

    @staticmethod
    def to_biluo(example: dict[str, list[int]]) -> dict[str, list[int]]:
        tags = [Label("GER").idx[tag] for tag in example["ner_tags"]]
        example["ner_tags"] = [Label("GER").labels[tag] for tag in iob_to_biluo(tags)]
        return example
