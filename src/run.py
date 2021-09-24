import pytorch_lightning as pl
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
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]
    return callbacks


def run(dataset, pl_model: pl.LightningModule) -> None:
    seed_everything(42, workers=True)

    datamodule: pl.LightningDataModule = DataModule(
        dataset=dataset,
        path=Path("data/distant_data/relations.csv"),
        num_workers=8,
        batch_size=16,
    )
    model: pl.LightningModule = pl_model()
    callbacks: list[Callback] = build_callbacks()
    csv_logger = CSVLogger(save_dir="csv_logs")

    # The Lightning core, the Trainer
    trainer: pl.Trainer = pl.Trainer(
        logger=[csv_logger],
        log_every_n_steps=10,
        callbacks=callbacks,
        deterministic=True,
        gpus=-1,
        precision=32,
        max_epochs=35,
        gradient_clip_val=0.5,
        auto_select_gpus=True,
        benchmark=True,
        amp_level="02",
        accumulate_grad_batches=1,
        stochastic_weight_avg=True,
        # auto_scale_batch_size="binsearch",
        fast_dev_run=False,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model)
    csv_logger.save()


if __name__ == "__main__":
    run(PandasDataset, RBERT)
