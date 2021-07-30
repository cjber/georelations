import hydra
import omegaconf
from omegaconf import ValueNode
import pandas as pd
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.common.model_utils import Const
from src.common.utils import PROJECT_ROOT


class TextDataset(Dataset):
    def __init__(
        self,
        path: ValueNode,
        coref_model: str,
        tokenizer=AutoTokenizer,
    ):
        self.data: pd.DataFrame = pd.read_csv(path)["text"]

        self.tokenizer = tokenizer.from_pretrained(
            Const.MODEL_NAME, add_prefix_space=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        text: str = self.data.iloc[index]
        encoding = self.tokenizer(
            text,
            return_attention_mask=True,
            max_length=Const.MAX_TOKEN_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: TextDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets, _recursive_=False
    )


if __name__ == "__main__":
    main()
