from typing import Any, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import AutoModel

from src.common.model_utils import Label
from src.common.utils import PROJECT_ROOT


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(pl.LightningModule):
    def __init__(self, model_name, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name

        self.model = AutoModel.from_pretrained(
            self.model_name,
            num_labels=Label("REL").count,
            return_dict=True,
            output_attentions=False,
        )

        hidden_size = self.model.config.hidden_size  # type:ignore
        dropout = self.model.config.hidden_dropout_prob  # type:ignore
        self.cls_fc_layer = FCLayer(hidden_size, hidden_size, dropout)
        self.entity_fc_layer = FCLayer(hidden_size, hidden_size, dropout)
        self.label_classifier = FCLayer(
            hidden_size * 3,
            Label("REL").count,
            dropout,
            use_activation=False,
        )

        self.train_f1 = pl.metrics.F1(num_classes=Label("REL").count)
        self.valid_f1 = pl.metrics.F1(num_classes=Label("REL").count)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask
    ):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        # Softmax
        if labels is not None:
            if Label("REL").count == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, Label("REL").count), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

    def step(self, batch: Any, batch_idx: int) -> dict:
        outputs = self(**batch)
        loss, logits = outputs[:2]
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        return {
            "probs": probs,
            "logits": logits,
            "loss": loss,
        }

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        step_out = self.step(batch, batch_idx)
        train_f1 = self.train_f1(step_out["probs"], batch["labels"])

        self.log("train_loss", step_out["loss"])
        self.log("train_f1", train_f1, prog_bar=True)
        return {"loss": step_out["loss"], "train_accuracy": train_f1}

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        step_out = self.step(batch, batch_idx)
        val_f1 = self.valid_f1(step_out["probs"], batch["labels"])

        self.log("val_loss", step_out["loss"])
        self.log("val_f1", val_f1)
        return {"val_loss": step_out["loss"], "val_accuracy": val_f1}

    def configure_optimizers(self) -> Union[Optimizer, dict]:
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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="rel")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
