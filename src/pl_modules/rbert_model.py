import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.common.utils import Const, Label, tdict
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import f1
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing import Any


class FCLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        use_activation: bool = True,
    ) -> None:
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.optim = AdamW
        self.scheduler = ReduceLROnPlateau

        self.model = AutoModel.from_pretrained(
            Const.MODEL_NAME,
            num_labels=Label("REL").count,
            return_dict=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            Const.MODEL_NAME, add_prefix_space=True
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": Const.SPECIAL_TOKENS}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

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
    def entity_average(
        hidden_output: Tensor,
        e_mask: Tensor,
    ) -> Tensor:
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
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        e1_mask: Tensor,
        e2_mask: Tensor,
    ) -> tdict:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # [CLS] token (bs, dim)

        # Average
        e1_h = self.entity_average(hidden_state, e1_mask)
        e2_h = self.entity_average(hidden_state, e2_mask)

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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, Label("REL").count), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

    def step(self, batch: tdict, _: int) -> tdict:
        outputs = self(**batch)
        loss, logits = outputs[:2]
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        return {"probs": probs, "logits": logits, "loss": loss}

    def training_step(self, batch: tdict, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        train_f1 = f1(
            step_out["probs"],
            batch["labels"],
            num_classes=Label("REL").count,
        )
        self.log_dict(
            {"train_loss": loss, "train_f1": train_f1},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: tdict, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        val_f1 = f1(
            step_out["probs"],
            batch["labels"],
            num_classes=Label("REL").count,
        )
        self.log_dict(
            {"val_loss": loss, "val_f1": val_f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: tdict, batch_idx: int) -> Tensor:
        step_out = self.step(batch, batch_idx)
        loss = step_out["loss"]
        test_f1 = f1(step_out["probs"], batch["labels"], num_classes=Label("REL").count)
        self.log_dict({"test_loss": loss, "test_f1": test_f1})
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

        opt = self.optim(lr=4e-5, params=optimizer_grouped_parameters)
        scheduler = self.scheduler(optimizer=opt, patience=2, verbose=True)

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "train_loss"}
