import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from src.pl_data.datamodule import DataModule
from src.pl_data.rel_dataset import RELDataset
from src.pl_data.wnut_dataset import WNUTDataset
from src.pl_module.ger_model import GERModel
from src.pl_module.rbert_model import RBERT
from typing import Union

DATA_DIR = Path("data")

parser = ArgumentParser()

parser.add_argument("--fast_dev_run", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", nargs="+", type=int, default=[42])
parser.add_argument("--model", type=str)
parser.add_argument("--save_to_hub", type=str)

args, unknown = parser.parse_known_args()

REL_DATA = (
    DATA_DIR / "rel_data" / "relations.csv"
    if not args.fast_dev_run
    else "tests/toy_data/train_rel.csv"
)
REL_DATA_TEST = (
    DATA_DIR / "rel_data" / "relations_test.csv"
    if not args.fast_dev_run
    else "tests/toy_data/train_rel.csv"
)


def build_callbacks() -> list[Callback]:
    callbacks: list[Callback] = [
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=True,
            min_delta=0.0,
            patience=3,
        ),
        ModelCheckpoint(
            filename="checkpoint",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
    ]
    return callbacks


def run(
    dataset,
    pl_model: pl.LightningModule,
    name: str,
    path: Union[Path, str],
    test_path: Union[Path, str],
    seed: int,
    args=args,
) -> None:
    seed_everything(seed, workers=True)

    datamodule: pl.LightningDataModule = DataModule(
        dataset=dataset,
        path=path,
        test_path=test_path,
        num_workers=8,
        batch_size=args.batch_size,
        seed=seed,
    )
    model: pl.LightningModule = pl_model()
    callbacks: list[Callback] = build_callbacks()
    csv_logger = CSVLogger(save_dir="csv_logs", name=f"seed_{seed}", version=name)

    if args.fast_dev_run:
        trainer_kwargs = {"gpus": None, "auto_select_gpus": False}
    else:
        trainer_kwargs = {"gpus": -1, "auto_select_gpus": True, "precision": 16}

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        **trainer_kwargs,
        deterministic=True,  # ensure reproducible results
        default_root_dir="ckpts",
        logger=[csv_logger],
        log_every_n_steps=10,
        callbacks=callbacks,
        max_epochs=35,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)

    if not args.fast_dev_run:
        test = trainer.test(model=model, ckpt_path="best", datamodule=datamodule)
        pd.DataFrame(test).to_csv(f"csv_logs/seed_{seed}_{name}_test.csv")
        csv_logger.save()

    if args.save_to_hub:
        model.model.push_to_hub(f"cjber/{args.save_to_hub}")  # type: ignore


if __name__ == "__main__":
    if len(args.seed) > 1:
        print("Running for multiple seeds.")
        for seed in args.seed:
            if args.model == "ger":
                run(
                    dataset=WNUTDataset,
                    pl_model=GERModel,
                    name="ger",
                    path="train",
                    test_path="test",
                    seed=seed,
                )
            elif args.model == "rel":
                run(
                    dataset=RELDataset,
                    pl_model=RBERT,
                    name="rel",
                    path=REL_DATA,
                    test_path=REL_DATA_TEST,
                    seed=seed,
                )
    elif args.model == "ger":
        run(
            dataset=WNUTDataset,
            pl_model=GERModel,
            name="ger",
            path="train",
            test_path="test",
            seed=args.seed[0],
        )
    elif args.model == "rel":
        run(
            dataset=RELDataset,
            pl_model=RBERT,
            name="rel",
            path=REL_DATA,
            test_path=REL_DATA_TEST,
            seed=args.seed[0],
        )
    else:
        print("Use valid --model arg [ger, rel].")
