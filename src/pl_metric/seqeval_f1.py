import torch
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU
from src.common.utils import Label
from torchmetrics import Metric


class Seqeval(Metric):
    def __init__(self, dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("f1s", default=torch.tensor([]))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        pred_biluo = []
        target_biluo = []
        for i, _ in enumerate(targets):
            true_labels_idx: list = [
                idx for idx, lab in enumerate(targets[i]) if lab != -100
            ]
            pred_biluo.append(
                [Label("GER").idx[pred.item()] for pred in preds[i, true_labels_idx]]
            )
            target_biluo.append(
                [
                    Label("GER").idx[target.item()]
                    for target in targets[i, true_labels_idx]
                ]
            )
        report: dict = classification_report(
            y_true=target_biluo,
            y_pred=pred_biluo,
            mode="strict",
            scheme=BILOU,
            output_dict=True,
            zero_division=1,
        )
        self.f1s = torch.cat(
            (
                self.f1s,
                torch.tensor([report.pop("micro avg")["f1-score"]], device=self.f1s.get_device()),
            )
        )

    def compute(self) -> dict:
        return self.f1s.mean()
