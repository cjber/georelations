from pathlib import Path
from src.pl_data.csv_dataset import RELDataset
from src.pl_data.wnut_dataset import WNUTDataset


class TestDatasets:
    def test_wnut(self):
        dataset: WNUTDataset = WNUTDataset("test")
        assert type(dataset) == WNUTDataset

    def test_csv(self):
        dataset: RELDataset = RELDataset(Path("tests/toy_data/train_rel.csv"))
        assert type(dataset) == RELDataset
