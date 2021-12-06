import os
import pandas as pd
from pathlib import Path
from src.common.utils import Const, Label, convert_input
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing import Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RELDataset(Dataset):
    def __init__(self, path: Path, tokenizer=AutoTokenizer) -> None:
        super().__init__()
        data: pd.DataFrame = pd.read_csv(path)
        data["relation"] = data["relation"].apply(lambda x: Label("REL").labels[x])
        tokenizer = tokenizer.from_pretrained(Const.MODEL_NAME)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )

        self.data: list[dict] = [
            convert_input(row, Const.MAX_TOKEN_LEN, tokenizer)
            for row in data.to_dict(orient="records")
        ]
        self.data = list(filter(None, self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Union[dict, None]:
        return self.data[index]
