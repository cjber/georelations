import pandas as pd
from pytorch_lightning.metrics import Metric
from seqeval.metrics import accuracy_score, classification_report
from seqeval.scheme import BILOU

from src.common.model_utils import Label


class Seqeval(Metric):
    def __init__(self):
        super().__init__()

        self.add_state("targets", default=[])
        self.add_state("preds", default=[])

        self.targets = []
        self.preds = []

    def update(self, preds, targets):
        self.preds.append(preds.flatten())
        self.targets.append(targets.flatten())

    def compute(self):
        pred_bioul = []
        target_bioul = []
        for pred_seq, target_seq in zip(self.preds, self.targets):
            pred_seq = pred_seq.cpu().numpy()
            target_seq = target_seq.squeeze().cpu().numpy()

            true_labels_idx: list = [
                idx
                for idx, lab in enumerate(target_seq)
                if lab != -100  # ignore special tags
            ]

            pred_bioul.append(
                [Label("GER").idx[pred] for pred in pred_seq[true_labels_idx]]
            )
            target_bioul.append(
                [Label("GER").idx[lab] for lab in target_seq[true_labels_idx]]
            )

        report: dict = classification_report(
            y_true=target_bioul,
            y_pred=pred_bioul,
            mode="strict",
            scheme=BILOU,
            output_dict=True,
            zero_division=1,  # no positive labels gives f1 of 1
        )  # type: ignore

        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")

        report["overall_precision"] = overall_score["precision"]
        report["overall_recall"] = overall_score["recall"]
        report["overall_f1"] = overall_score["f1-score"]
        report["overall_accuracy"] = accuracy_score(
            y_true=target_bioul, y_pred=pred_bioul
        )

        return pd.json_normalize(report, sep="_").to_dict(orient="records")[0]
