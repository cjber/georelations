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
from src.pl_data.csv_dataset import RELDataset
from src.pl_data.datamodule import DataModule
from src.pl_data.wnut_dataset import WNUTDataset
from src.pl_modules.ger_model import GERModel
from src.pl_modules.rbert_model import RBERT
from typing import Union

DATA_DIR = Path("data")

parser = ArgumentParser()

parser.add_argument("--fast_dev_run", type=bool, default=False)
parser.add_argument("--seed", nargs="+", type=int, default=[42])
parser.add_argument("--model", type=str)
parser.add_argument("--save_to_hub", type=str)

args = parser.parse_args()


def build_callbacks() -> list[Callback]:
    callbacks: list[Callback] = [
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
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
) -> None:  # sourcery skip: boolean-if-exp-identity
    seed_everything(seed, workers=True)

    datamodule: pl.LightningDataModule = DataModule(
        dataset=dataset,
        path=path,
        test_path=test_path,
        num_workers=8,
        batch_size=2,
        seed=seed,
    )
    model: pl.LightningModule = pl_model()
    callbacks: list[Callback] = build_callbacks()
    csv_logger = CSVLogger(
        save_dir="csv_logs",
        name="seed_" + str(seed),
        version=name,
    )

    gpus = None if args.fast_dev_run else -1
    auto_select_gpus = False if args.fast_dev_run else True
    precision = 64 if args.fast_dev_run else 16

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        deterministic=True,  # ensure reproducible results
        default_root_dir="ckpts",
        logger=[csv_logger],
        log_every_n_steps=10,
        callbacks=callbacks,
        gpus=gpus,
        auto_select_gpus=auto_select_gpus,
        precision=precision,
        max_epochs=35,
        benchmark=True,
        stochastic_weight_avg=True,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model)

    if not args.fast_dev_run:
        test = trainer.test()
        pd.DataFrame(test).to_csv(
            "csv_logs/seed_" + str(seed) + "_" + name + "_test.csv"
        )
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
                    path=DATA_DIR / "distant_data" / "relations.csv",
                    test_path=DATA_DIR / "distant_data" / "relations_test.csv",
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
            path=DATA_DIR / "distant_data" / "relations.csv",
            test_path=DATA_DIR / "distant_data" / "relations_test.csv",
            seed=args.seed[0],
        )
    else:
        print("Use valid --model arg [ger, rel].")
