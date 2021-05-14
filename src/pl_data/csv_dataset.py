import hydra
import omegaconf
from omegaconf import ValueNode
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.common.model_utils import Config, convert_examples_to_features
from src.common.utils import PROJECT_ROOT



class CSVDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        tokenizer=AutoTokenizer,
    ):
        super().__init__()
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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: CSVDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
