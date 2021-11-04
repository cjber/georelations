import pandas as pd
from pathlib import Path
from src.common.utils import Const, Label, convert_examples_to_features
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing import Union


class RELDataset(Dataset):
    def __init__(self, path: Path, tokenizer=AutoTokenizer) -> None:
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(path)
        self.data["relation"] = self.data["relation"].apply(
            lambda x: Label("REL").labels[x]
        )

        self.tokenizer = tokenizer.from_pretrained(Const.MODEL_NAME)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Union[dict, None]:
        item = self.data.iloc[index].to_dict()
        return convert_examples_to_features(
            item,
            max_seq_len=Const.MAX_TOKEN_LEN,
            tokenizer=self.tokenizer,
        )
