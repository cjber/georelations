# https://github.com/Spico197/NYT-H

import jsonlines
import numpy as np
import pandas as pd
from src.common.utils import Const


def process_relations(relations):
    return [
        {
            "head": line["head"]["word"],
            "relation": line["relation"].split("/")[-1],
            "tail": line["tail"]["word"],
            "sentence": line["sentence"],
        }
        for line in relations
    ]


def add_tag(sentence, tag_idx, tag_length, tag_name):
    return (
        sentence[:tag_idx]
        + f"<{tag_name}>"
        + sentence[tag_idx : tag_idx + tag_length]
        + f"</{tag_name}>"
        + sentence[tag_idx + tag_length :]
    )


def create_df(item):
    relation = item["relation"]
    sentence = item["sentence"]
    head = item["head"]
    tail = item["tail"]
    head_len = len(head)
    tail_len = len(tail)

    head_idx = sentence.find(head + " ")
    sentence = add_tag(sentence, head_idx, head_len, "head")
    tail_idx = sentence.find(tail + " ")

    if tail_idx == -1:
        sentence = item["sentence"]  # reset sentence
        head_idx = sentence.rfind(head + " ")
        sentence = add_tag(sentence, head_idx, head_len, "head")
        tail_idx = sentence.find(tail + " ")

    if not (tail_idx < head_idx) & (tail_idx > head_idx + head_len):
        tail_idx = sentence.rfind(tail + " ")

    sentence = add_tag(sentence, tail_idx, tail_len, "tail")

    sentence = sentence[: Const.MAX_TOKEN_LEN]

    if any(x not in sentence for x in Const.SPECIAL_TOKENS):
        return {"relation": np.nan, "sentence": np.nan}

    assert all(x in sentence for x in Const.SPECIAL_TOKENS)
    assert all(x not in sentence for x in ["<head></head>", "<tail></tail>"])

    relation = relation if relation != "NA" else "none"
    return {"relation": relation, "sentence": sentence}


if __name__ == "__main__":
    with jsonlines.open("./data/distant_data/train.json", "r") as jl:
        relations = [
            line
            for line in jl
            if ("location" in line["head"]["type"])
            & ("location" in line["tail"]["type"])
            & (line["relation"].startswith("/location/") | (line["relation"] == "NA"))
        ]

    output = process_relations(relations)

    out = (
        pd.DataFrame([create_df(item) for item in output])
        .dropna()
        .drop_duplicates()
        .loc[lambda x: x["relation"].isin(["contains", "none"])]  # type: ignore
    )

    out.to_csv("./data/distant_data/relations.csv", index=False)
