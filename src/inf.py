import hydra
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig, ValueNode
from src.common.utils import PROJECT_ROOT, load_envs
from src.pl_data.text_dataset import TextDataset
from src.pl_modules.ensemble import RelationEnsemble
from src.pl_modules.ger_model import GERModel
from src.pl_modules.rbert_model import RBERT
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

pl.seed_everything(42)
torch.use_deterministic_algorithms(True)
load_envs()


def load_pretrained_model(model, checkpoint, device):
    model = model.load_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()
    model.freeze()
    return model


def load_dataset(datadir: ValueNode, coref_model: str):
    ds = TextDataset(path=datadir, coref_model=coref_model)
    return DataLoader(dataset=ds, batch_size=64, shuffle=True)


def run():
    rel_model = load_pretrained_model(RBERT, "", "cuda")

    model = RelationEnsemble(ger_model, rel_model)
    model = model.to("cuda")

    dl = load_dataset(
        datadir=cfg.data.datamodule.datasets.path, coref_model="coref-spanbert"
    )
    preds = [
        model(item["input_ids"].to("cuda"), item["attention_mask"].to("cuda"))
        for item in tqdm(dl)
    ]

    triples = []
    for pred in preds:
        text = pred[0][0]
        rel = pred[0][1]

        e1 = text.split("<head>")[1].split("</head>")[0].strip()
        e2 = text.split("<tail>")[1].split("</tail>")[0].strip()
        triple = (e1, e2, rel)
        triples.append(triple)

    breakpoint()


if __name__ == "__main__":
    run()
