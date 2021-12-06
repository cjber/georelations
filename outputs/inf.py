import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from src.pl_data.jsonl_dataset import JSONLDataset
from src.pl_module.ensemble import RelationEnsemble
from src.pl_module.ger_model import GERModel
from src.pl_module.rbert_model import RBERT
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

pl.seed_everything(42)


def load_pretrained_model(model, checkpoint, device):
    model = model.load_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()
    model.freeze()
    return model


def load_dataset(datadir: Path):
    ds = JSONLDataset(path=datadir)
    return DataLoader(dataset=ds, batch_size=64)


def run():
    ger_model = load_pretrained_model(
        GERModel, "ckpts/seed_42/ger/checkpoints/checkpoint.ckpt", "cuda"
    )
    rel_model = load_pretrained_model(
        RBERT, "ckpts/seed_42/rel/checkpoints/checkpoint.ckpt", "cuda"
    )

    model = RelationEnsemble(ger_model, rel_model)
    model = model.to("cuda")

    dl = load_dataset(datadir=Path("data/reddit_comments/liverpool.jsonl"))
    preds = [
        model(item["input_ids"].to("cuda"), item["attention_mask"].to("cuda"))
        for item in tqdm(dl)
    ]

    triples = []
    preds = filter(None, [item for sublist in preds for item in sublist])
    for pred in preds:
        text = pred[0]
        rel = pred[1]

        head = text.split("<head>")[1].split("</head>")[0].strip()
        tail = text.split("<tail>")[1].split("</tail>")[0].strip()
        triple = {"head": head, "rel": rel, "tail": tail, "text": text}
        triples.append(triple)
    pd.DataFrame(triples).to_csv("data/out/triples.csv", index=False)


if __name__ == "__main__":
    run()
