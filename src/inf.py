import hydra
from omegaconf.dictconfig import DictConfig, ValueNode
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.common.utils import load_envs
from src.common.utils import PROJECT_ROOT
from src.pl_data.text_dataset import TextDataset
from src.pl_modules.ensemble import RelationEnsemble
from src.pl_modules.ger_model import GERModel
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


def load_dataset(datadir: ValueNode, coref_model: str):
    ds = TextDataset(path=datadir, coref_model=coref_model)
    return DataLoader(dataset=ds, batch_size=64, shuffle=True)


def run(cfg: DictConfig):
    ger_model = load_pretrained_model(GERModel, cfg.model.ger_model, "cuda")
    rel_model = load_pretrained_model(RBERT, cfg.model.rel_model, "cuda")

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

        e1 = text.split("<e1>")[1].split("</e1>")[0].strip()
        e2 = text.split("<e2>")[1].split("</e2>")[0].strip()
        triple = (e1, e2, rel)
        triples.append(triple)

    breakpoint()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="ens")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
