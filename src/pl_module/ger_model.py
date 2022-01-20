import pytorch_lightning as pl
import torch
from src.common.utils import Const, Label, combine_biluo, combine_subwords, tdict
from src.pl_metric.seqeval_f1 import Seqeval
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import Any, Union


class GERModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.optim = AdamW
        self.scheduler = ReduceLROnPlateau

        self.train_f1 = Seqeval()
        self.val_f1 = Seqeval()
        self.test_f1 = Seqeval()

        self.model = AutoModelForTokenClassification.from_pretrained(
            Const.MODEL_NAME,
            num_labels=Label("GER").count,
            return_dict=True,
            id2label=Label("GER").idx,
            label2id=Label("GER").labels,
            finetuning_task="ger",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            Const.MODEL_NAME, add_prefix_space=True
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
    ) -> Union[tdict, list[tdict]]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits loss)
        """
        outputs: tdict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if labels is None:
            batch_out = []
            for id_batch, logit_batch in zip(input_ids, outputs["logits"]):
                tokens: list = self.tokenizer.convert_ids_to_tokens(id_batch.tolist())
                tags: list = logit_batch.argmax(dim=1).tolist()
                tokens, tags = combine_subwords(tokens, tags)
                if tokens:
                    tokens, tags = combine_biluo(tokens, tags)
                    batch_out.append({"tokens": tokens, "tags": tags})
            return batch_out
        return outputs

    def step(self, batch: tdict, _: int) -> tdict:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {
            "preds": outputs["logits"].argmax(dim=2),
            "loss": outputs["loss"],
        }

    def training_step(self, batch: tdict, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        self.train_f1(preds=step_out["preds"], targets=batch["labels"])
        self.log_dict({"train_loss": loss, "train_f1": self.train_f1}, prog_bar=False)
        return loss

    def validation_step(self, batch: tdict, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        self.val_f1(preds=step_out["preds"], targets=batch["labels"])
        self.log_dict({"val_loss": loss, "val_f1": self.val_f1}, prog_bar=True)
        return loss

    def test_step(self, batch: tdict, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        self.test_f1(step_out["preds"], batch["labels"])
        self.log_dict({"test_loss": loss, "test_f1": self.test_f1})
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if all(nd not in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        opt = self.optim(lr=2e-5, params=optimizer_grouped_parameters)
        scheduler = self.scheduler(optimizer=opt, patience=1, verbose=True, mode="min")

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
