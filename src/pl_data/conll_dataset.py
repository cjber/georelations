import itertools
import torch
from src.common.utils import Const, Label, encode_labels
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class CoNLLDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        tokenizer=AutoTokenizer,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.model_name = Const.MODEL_NAME
        self.max_token_len = Const.MAX_TOKEN_LEN

        self.data: list[dict[str, tuple[str]]] = self.read_conll()
        self.tokenizer = tokenizer.from_pretrained(
            self.model_name, add_prefix_space=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        tokens: list[str] = list(self.data[index]["tokens"])
        labels_id: list[int] = [
            Label("GER").labels[label] for label in self.data[index]["tags"]
        ]
        encoding, labels_encoded = encode_labels(
            tokens, labels_id, self.tokenizer, self.max_token_len
        )
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.LongTensor(labels_encoded),
        )

    def __repr__(self) -> str:
        return (
            f"CoNLLDataset("
            f"{self.name=}, "
            f"{self.path=}, "
            f"{self.model_name=}, "
            f"{self.max_token_len=}, "
            f"{self.tokenizer=})"
        )

    def read_conll(self) -> list[dict[str, tuple[str]]]:
        data: list[dict[str, tuple[str]]] = []
        # with open(self.path, "r") as conll_file:  # type: ignore
        # for divider, lines in itertools.groupby(
        #     conll_file, lambda x: x.startswith("-DOCSTART-")
        # ):
        #     if divider:
        #         continue
        #     fields = [line.strip().split() for line in lines]
        #     fields = [line for line in fields if line]
        #     fields = [line for line in zip(*fields)]
        #     tokens, _, _, tags = fields

        with open(self.path, "r") as conll_file:
            for divider, lines in itertools.groupby(
                conll_file, lambda x: x.strip() == ""
            ):
                if divider:
                    continue
                fields = [line.strip().split() for line in lines]
                fields = [line for line in zip(*fields)]
                tokens, ner_tags = fields
                ner_tags = ["O" if tag[-3:] == "NOM" else tag for tag in ner_tags]
                ner_tags = [tag[:-4] if tag != "O" else tag for tag in ner_tags]

                sequence = {"tokens": tokens, "tags": ner_tags}
                data.append(sequence)
        return data


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="ger")
def main(cfg: omegaconf.DictConfig):
    dataset: CoNLLDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
