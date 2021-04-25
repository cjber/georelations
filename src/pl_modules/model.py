from typing import Any, Union

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.optim import Optimizer
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.common.utils import Config, combine_biluo, combine_subwords
from src.pl_metric.metric import Seqeval


class GERModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.train_metric = Seqeval()
        self.val_metric = Seqeval()

        self.model = AutoModelForTokenClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=Config.GER_NUM,
            return_dict=True,
            output_attentions=False,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits loss)
        """
        outputs: dict = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        if labels is None:
            tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME, add_prefix_space=True
            )
            batch_out = []
            for id_batch, logit_batch in zip(input_ids, outputs["logits"]):
                tokens: list = tokenizer.convert_ids_to_tokens(  # type:ignore
                    id_batch.tolist()
                )
                tags: list = logit_batch.argmax(dim=1).tolist()
                tokens, tags = combine_subwords(tokens, tags)
                tokens, tags = combine_biluo(tokens, tags)
                batch_out.append({"tokens": tokens, "tags": tags})
            return batch_out
        return outputs

    def step(self, batch: Any, batch_idx: int) -> dict:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {
            "preds": outputs["logits"].argmax(dim=2),
            "loss": outputs["loss"],
        }

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        step_out = self.step(batch, batch_idx)
        train_metrics = self.train_metric(
            preds=step_out["preds"], targets=batch["labels"]
        )
        train_f1 = train_metrics.pop("overall_f1")

        self.log("train_loss", step_out["loss"])
        self.log("train_f1", train_f1, prog_bar=True)
        return {
            "loss": step_out["loss"],
            "train_f1": train_f1,
            "train_metrics": train_metrics,
        }

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        step_out = self.step(batch, batch_idx)
        val_metrics = self.val_metric(preds=step_out["preds"], targets=batch["labels"])
        val_f1 = val_metrics.pop("overall_f1")
        self.log("val_loss", step_out["loss"])
        self.log("val_f1", val_f1, prog_bar=True)
        self.log("val", val_metrics)
        return {
            "val_loss": step_out["loss"],
            "val_f1": val_f1,
            "val_metrics": val_metrics,
        }

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, dict]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of
                          LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
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
        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer, params=optimizer_grouped_parameters
        )

        if self.cfg.optim.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.cfg.optim.lr_scheduler, optimizer=opt
            )
            return {
                "optimizer": opt,
                "lr_scheduler": scheduler,
                "monitor": self.cfg.train.monitor_metric,
            }

        return opt
