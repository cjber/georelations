import itertools

from allennlp_models.pretrained import load_predictor
import hydra
import omegaconf
from omegaconf import DictConfig, ValueNode
import pandas as pd
from spacy.training import iob_to_biluo
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.common.model_utils import Config, convert_examples_to_features, encode_labels
from src.common.utils import PROJECT_ROOT


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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: TextDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
