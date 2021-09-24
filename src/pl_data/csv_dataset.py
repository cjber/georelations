import pandas as pd
from pathlib import Path
from src.common.utils import Const, Label, convert_examples_to_features
from torch.utils.data import Dataset
from transformers import AutoTokenizer  # type: ignore


class PandasDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer=AutoTokenizer,
    ):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(path)
        self.data["relation"] = self.data["relation"].apply(
            lambda x: Label("REL").labels[x]
        )

        self.tokenizer = tokenizer.from_pretrained(Const.MODEL_NAME)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )

        self.labels = self.data["relation"].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data.iloc[index]
        input = convert_examples_to_features(
            item, max_seq_len=Const.MAX_TOKEN_LEN, tokenizer=self.tokenizer
        )

        input["text"] = item["sentence"]
        return input
