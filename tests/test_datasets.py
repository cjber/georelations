import hydra
from hydra.experimental import compose
from hydra.experimental.initialize import initialize

from src.pl_data.conll_dataset import CoNLLDataset
from src.pl_data.csv_dataset import CSVDataset
from src.pl_modules.ger_model import GERModel


class TestDatasets:
    def test_conll(self):
        with initialize(config_path="../conf", job_name="test"):
            cfg = compose(config_name="ger")
            dataset: CoNLLDataset = hydra.utils.instantiate(
                cfg.data.datamodule.datasets.train, _recursive_=False
            )
            assert type(dataset) == CoNLLDataset

    def test_csv(self):
        with initialize(config_path="../conf", job_name="test"):
            cfg = compose(config_name="rel")
            dataset: CSVDataset = hydra.utils.instantiate(
                cfg.data.datamodule.datasets.train, _recursive_=False
            )
            assert type(dataset) == CSVDataset


class TestModules:
    def test_ger(self):
        with initialize(config_path="../conf", job_name="test"):
            cfg = compose(config_name="ger")
            model: GERModel = hydra.utils.instantiate(
                cfg.model,
                optim=cfg.optim,
                data=cfg.data,
                logging=cfg.logging,
                _recursive_=False,
            )
            assert type(model) == GERModel
