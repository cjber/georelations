import itertools
import pytorch_lightning as pl
import torch
from src.common.utils import Const, Label, convert_input, ents_to_relations
from transformers import AutoTokenizer


class RelationEnsemble(pl.LightningModule):
    def __init__(
        self,
        ger_model: pl.LightningModule,
        rel_model: pl.LightningModule,
        tokenizer=AutoTokenizer,
    ) -> None:
        super().__init__()

        self.ger_model = ger_model
        self.rel_model = rel_model

        self.tokenizer = tokenizer.from_pretrained(Const.MODEL_NAME)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels=None,
    ) -> list[tuple]:
        ger_out: dict = self.ger_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        sequence: list = [
            ents_to_relations(item["tokens"], item["tags"]) for item in ger_out
        ]
        sequence = list(itertools.chain.from_iterable([s for s in sequence if s]))

        output = []
        if sequence:
            for text in sequence:
                if any(
                    x not in text for x in ["<head>", "</head>", "<tail>", "</tail>"]
                ):
                    continue
                item = {}
                item["sentence"] = text

                features = convert_input(
                    item,
                    max_seq_len=Const.MAX_TOKEN_LEN,
                    tokenizer=self.tokenizer,
                )

                # truncation may remove tail/head so ignore
                if features:
                    rel_out = self.rel_model(
                        features["input_ids"].unsqueeze(0).to("cuda"),
                        features["attention_mask"].unsqueeze(0).to("cuda"),
                        features["labels"],
                        features["e1_mask"].unsqueeze(0).to("cuda"),
                        features["e2_mask"].unsqueeze(0).to("cuda"),
                    )

                    output.append(
                        (item, Label("REL").idx[rel_out[0].argmax(dim=1)[0].tolist()])
                    )
        return output
