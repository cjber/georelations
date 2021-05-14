import itertools

import pytorch_lightning as pl
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.common.model_utils import (
    Config,
    convert_examples_to_features,
    ents_to_relations,
)


class RelationEnsemble(pl.LightningModule):
    def __init__(
        self, ger_model, rel_model, tokenizer=AutoTokenizer, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.ger_model = ger_model
        self.rel_model = rel_model

        ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer = tokenizer.from_pretrained(Config.MODEL_NAME)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}
        )

    def forward(self, input_ids, attention_mask, labels=None):
        ger_out: dict = self.ger_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        sequence_list = [
            ents_to_relations(item["tokens"], item["tags"]) for item in ger_out
        ]
        sequence_list = list(
            itertools.chain.from_iterable([s for s in sequence_list if s])
        )

        if sequence_list:
            features = [
                convert_examples_to_features(
                    item,
                    max_seq_len=Config.MAX_TOKEN_LEN,
                    tokenizer=self.tokenizer,
                    labels=False,
                )
                for item in sequence_list
            ]
            rel_in = {
                "input_ids": torch.tensor(
                    [f.input_ids for f in features], dtype=torch.long, device="cuda"
                ),
                "attention_mask": torch.tensor(
                    [f.attention_mask for f in features],
                    dtype=torch.long,
                    device="cuda",
                ),
                "token_type_ids": torch.tensor(
                    [f.token_type_ids for f in features],
                    dtype=torch.long,
                    device="cuda",
                ),
                "e1_mask": torch.tensor(
                    [f.e1_mask for f in features], dtype=torch.long, device="cuda"
                ),
                "e2_mask": torch.tensor(
                    [f.e2_mask for f in features], dtype=torch.long, device="cuda"
                ),
            }
            rel_out = self.rel_model(**rel_in, labels=None)
            return (
                sequence_list,
                [Config.REL_IDX[rel] for rel in rel_out[0].argmax(dim=1).tolist()],
            )
