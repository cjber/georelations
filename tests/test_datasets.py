from src.pl_data.conll_dataset import CoNLLDataset
from src.pl_data.csv_dataset import RELDataset
from src.pl_data.datamodule import DataModule
from src.pl_modules.ger_model import GERModel


class TestDatasets:
    def test_conll(self):
        dataset = CoNLLDataset("../tests/toy_data/train_ger.conll")
        assert type(dataset) == CoNLLDataset

    def test_csv(self):
            dataset: RELDataset = 
                "../tests/toy_data/train_rel.csv", _recursive_=False
            )
            assert type(dataset) == RELDataset


class TestModules:
    def test_datamodule(self):
        with initialize(config_path="../conf", job_name="test"):
            cfg = compose(config_name="ger")
            datamodule: DataModule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False
            )
            assert type(datamodule) == DataModule


class TestModels:
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
