from ast import literal_eval

import hydra
import omegaconf
from omegaconf import ValueNode
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.common.model_utils import Label, convert_examples_to_features
from src.common.utils import PROJECT_ROOT


class CSVDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        model_name: ValueNode,
        max_token_len: ValueNode,
        special_tokens: ValueNode,
        tokenizer=AutoTokenizer,
    ):
        super().__init__()
        self.name = name
        self.data: pd.DataFrame = pd.read_csv(
            path, header=None, names=["label", "text_a"]  # type: ignore
        )
        self.model_name = model_name
        self.max_token_len = int(max_token_len)
        self.special_tokens: list[str] = literal_eval(special_tokens)

        self.tokenizer = tokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.special_tokens}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data.iloc[index]

        item.label = Label("REL").labels[item.label]
        return convert_examples_to_features(
            item, max_seq_len=self.max_token_len, tokenizer=self.tokenizer
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: CSVDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
