import jsonlines
import torch
from ekphrasis.classes.preprocessor import TextPreProcessor
from pathlib import Path
from src.common.utils import Const
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


class JSONLDataset(Dataset):
    def __init__(
        self, path: Path, tokenizer: PreTrainedTokenizer = AutoTokenizer  # type: ignore
    ) -> None:
        with jsonlines.open(path, "r") as jl:
            self.data = [line["text"] for line in jl]  # type: ignore (jsonlines issue)
        self.text_processor = TextPreProcessor(**Const.TEXT_PROCESSOR_ARGS)

        self.data = [self.normalise(line) for line in self.data]
        self.tokenizer = tokenizer.from_pretrained(
            Const.MODEL_NAME,
            add_prefix_space=True,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text: str = self.data[index]
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

    def normalise(self, line: str) -> str:
        return " ".join(self.text_processor.pre_process_doc(line))
