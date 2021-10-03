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
from src.pl_data.csv_dataset import PandasDataset
from src.pl_data.datamodule import DataModule
from src.pl_modules.rbert_model import RBERT

parser = ArgumentParser()

parser.add_argument("--fast_dev_run", type=bool, default=False)
parser.add_argument("--seed", nargs="+", type=int, default=[42])

args = parser.parse_args()


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
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]
    return callbacks


def run(dataset, pl_model: pl.LightningModule, seed, args=args) -> None:

    seed_everything(seed, workers=True)

    datamodule: pl.LightningDataModule = DataModule(
        dataset=dataset,
        path=Path("data/distant_data/relations.csv"),
        num_workers=8,
        batch_size=16,
    )
    model: pl.LightningModule = pl_model()
    callbacks: list[Callback] = build_callbacks()
    csv_logger = CSVLogger(
        save_dir="csv_logs",
        name="seed_" + str(seed),
        version=0,
    )

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        deterministic=True,  # ensure reproducible results
        default_root_dir="ckpts",
        logger=[csv_logger],
        log_every_n_steps=10,
        callbacks=callbacks,
        gpus=-1,
        precision=16,
        max_epochs=35,
        auto_select_gpus=True,
        benchmark=True,
        stochastic_weight_avg=True,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model)
    csv_logger.save()


if __name__ == "__main__":
    if len(args.seed) > 1:
        print("Running for multiple seeds.")
        for seed in args.seed:
            run(PandasDataset, RBERT, seed=seed)
    else:
        run(PandasDataset, RBERT, seed=args.seed[0])
