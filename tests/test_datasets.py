from pathlib import Path
from src.datasets.datamodule import DataModule
from src.datasets.rel_dataset import RELDataset
from src.datasets.wnut_dataset import WNUTDataset
from src.modules.ger_model import GERModel


class TestDatasets:
    @staticmethod
    def test_wnut():
        dataset: WNUTDataset = WNUTDataset("test")
        assert type(dataset) == WNUTDataset

    @staticmethod
    def test_rel():
        dataset: RELDataset = RELDataset(Path("tests/toy_data/train_rel.csv"))
        assert type(dataset) == RELDataset


class TestModules:
    @staticmethod
    def test_wnut():
        datamodule: DataModule = DataModule(
            dataset=WNUTDataset,
            path="test",
            test_path="test",
            num_workers=1,
            batch_size=1,
            seed=1,
        )
        assert type(datamodule) == DataModule

    @staticmethod
    def test_rel():
        datamodule: DataModule = DataModule(
            dataset=RELDataset,
            path=Path("tests/toy_data/train_rel.csv"),
            test_path=Path("tests/toy_data/train_rel.csv"),
            num_workers=1,
            batch_size=1,
            seed=1,
        )
        assert type(datamodule) == DataModule


class TestModels:
    @staticmethod
    def test_ger():
        model: GERModel = GERModel()
        assert type(model) == GERModel
