import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.common.utils import load_envs
from src.pl_data.dataset import TextDataset
from src.pl_modules.ensemble import RelationEnsemble
from src.pl_modules.model import GERModel
from src.pl_modules.rbert_model import RBERT


pl.seed_everything(42)
torch.use_deterministic_algorithms(True)
load_envs()


def load_pretrained_model(model, checkpoint, device):
    model = model.load_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()
    model.freeze()
    return model


def load_dataset(datadir: str, coref_model: str):
    text = []
    with open(datadir, "r") as f:
        for line in f:
            text.append(line.strip())
    ds = TextDataset(text=text, coref_model=coref_model)
    return DataLoader(dataset=ds, batch_size=64, shuffle=True)


def run():
    ger_model = load_pretrained_model(GERModel, "models/ger_model.ckpt", "cuda")
    rel_model = load_pretrained_model(RBERT, "models/rel_model.ckpt", "cuda")

    model = RelationEnsemble(ger_model, rel_model)
    model = model.to("cuda")

    dl = load_dataset(
        datadir="labelling/ldata/wiki_corefs.txt", coref_model="coref-spanbert"
    )
    preds = [
        model(item["input_ids"].to("cuda"), item["attention_mask"].to("cuda"))
        for item in tqdm(dl)
    ]

    text = preds[0][0][5]
    rel = preds[0][1][5]
    e1 = text.split("<e1>")[1].split("</e1>")[0].strip()
    e2 = text.split("<e2>")[1].split("</e2>")[0].strip()
    triple = (e1, e2, rel)
