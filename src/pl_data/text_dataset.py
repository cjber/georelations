from allennlp_models.pretrained import load_predictor
import hydra
import omegaconf
from omegaconf import ValueNode
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from common.model_utils import Const
from src.common.utils import PROJECT_ROOT


class TextDataset(Dataset):
    def __init__(
        self,
        path: ValueNode,
        coref_model: str,
        tokenizer=AutoTokenizer,
    ):
        predictor = load_predictor(coref_model)
        predictor.cuda_device = 0 if torch.cuda.is_available else -1

        self.text: list[str] = []
        with open(path, "r") as f:
            for line in f:
                self.text.append(line.strip())
        self.text = [predictor.coref_resolved(i) for i in tqdm(self.text)]

        self.tokenizer = tokenizer.from_pretrained(
            Const.MODEL_NAME, add_prefix_space=True
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index: int):
        text: str = self.text[index]
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
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
