import jsonlines
import numpy as np
import pandas as pd
from pathlib import Path
from src.common.utils import Const

# https://github.com/Spico197/NYT-H
from typing import Union


class NYTReader:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path
        with jsonlines.open(self.path, "r") as file:
            self.relations = [line for line in file]  # type: ignore

    def read(self) -> pd.DataFrame:
        self.get_locations()
        self.process_relations()
        self.process_sentence()

        return (
            pd.DataFrame(self.relations)
            .dropna()
            .drop_duplicates()
            .drop("bag_label", axis=1)
        )

    def get_locations(self):
        self.relations = [
            line
            for line in self.relations
            if ("location" in line["head"]["type"])
            & ("location" in line["tail"]["type"])
            & (line["relation"].startswith("/location/") | (line["relation"] == "NA"))
        ]

    def process_relations(self):
        self.relations = [
            {
                "head": line["head"]["word"],
                "relation": line["relation"].split("/")[-1],
                "tail": line["tail"]["word"],
                "sentence": line["sentence"],
                "bag_label": line["bag_label"],
            }
            for line in self.relations
        ]

    @staticmethod
    def add_tag(sentence, tag_idx, tag_length, tag_name):
        return (
            sentence[:tag_idx]
            + f"<{tag_name}>"
            + sentence[tag_idx : tag_idx + tag_length]
            + f"</{tag_name}>"
            + sentence[tag_idx + tag_length :]
        )

    def process_sentence(self):
        relations = []
        for item in self.relations:
            relation = item["relation"]
            sentence = item["sentence"]
            head = item["head"]
            tail = item["tail"]
            head_len = len(head)
            tail_len = len(tail)

            # only keep relevant relationships
            if relation not in ["neighbourhood_of", "contains"]:
                relation = "none"
            # bag label = no means wrong relationship
            if item["bag_label"] not in ["unk", "yes"]:
                relation = "none"
            # relationship is reversed
            if relation == "neighbourhood_of":
                head, tail = tail, head

            head_idx = sentence.find(head + " ")
            sentence = self.add_tag(sentence, head_idx, head_len, "head")
            tail_idx = sentence.find(tail + " ")

            # if tail not found in sentence, reset sentence and reverse head search
            if tail_idx == -1:
                sentence = item["sentence"]
                head_idx = sentence.rfind(head + " ")
                sentence = self.add_tag(sentence, head_idx, head_len, "head")
                tail_idx = sentence.find(tail + " ")

            # if tail is found within head, reverse tail search
            if not (tail_idx < head_idx) & (tail_idx > head_idx + head_len):
                tail_idx = sentence.rfind(tail + " ")

            sentence = self.add_tag(sentence, tail_idx, tail_len, "tail")
            sentence = sentence[: Const.MAX_TOKEN_LEN]

            if any(
                x not in sentence for x in ["<head>", "</head>", "<tail>", "</tail>"]
            ):
                relations.append(
                    {"relation": np.nan, "sentence": np.nan, "bag_label": np.nan}
                )
            else:
                # ensure no empty entities
                assert all(
                    x not in sentence for x in ["<head></head>", "<tail></tail>"]
                )
                relations.append(
                    {
                        "relation": relation,
                        "sentence": sentence,
                        "bag_label": item["bag_label"],
                    }
                )
        self.relations = relations


def main():
    reader = NYTReader(Path("data/distant_data/train.json"))
    relations = reader.read()
    relations.to_csv("./data/distant_data/relations.csv", index=False)

    reader = NYTReader(Path("data/distant_data/test.json"))
    relations_test = reader.read()
    relations_test.to_csv("./data/distant_data/relations_test.csv", index=False)


if __name__ == "__main__":
    main()
