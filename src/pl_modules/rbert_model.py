import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.common.utils import Const, Label
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import f1
from transformers import AutoModel
from typing import Any, Union


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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.optim = AdamW
        self.scheduler = ReduceLROnPlateau

        self.model = AutoModel.from_pretrained(
            Const.MODEL_NAME,
            num_labels=Label("REL").count,
            return_dict=True,
            # output_hidden_states=True,
            output_attentions=False,
        )

        hidden_size = self.model.config.hidden_size
        dropout = (
            0
            if Const.MODEL_NAME.startswith("distil")
            else self.model.config.hidden_dropout_prob
        )
        self.cls_fc_layer = FCLayer(hidden_size, hidden_size, dropout)
        self.entity_fc_layer = FCLayer(hidden_size, hidden_size, dropout)
        self.label_classifier = FCLayer(
            hidden_size * 3,
            Label("REL").count,
            dropout,
            use_activation=False,
        )

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
        return sum_vector.float() / length_tensor.float()

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, text
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[0][:, 0]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat(
            [
                pooled_output,
                e1_h,
                e2_h,
            ],
            dim=-1,
        )
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
        train_f1 = f1(
            step_out["probs"],
            batch["labels"],
            num_classes=Label("REL").count,
        )

        self.log("train_loss", step_out["loss"])
        self.log("train_f1", train_f1)
        return {"loss": step_out["loss"], "train_accuracy": train_f1}

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        step_out = self.step(batch, batch_idx)
        val_f1 = f1(step_out["probs"], batch["labels"], num_classes=Label("REL").count)

        self.log("val_loss", step_out["loss"])
        self.log("val_f1", val_f1, prog_bar=True)
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

        opt = self.optim(lr=4e-5, params=optimizer_grouped_parameters)
        scheduler = self.scheduler(optimizer=opt, patience=2, verbose=True)

        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
