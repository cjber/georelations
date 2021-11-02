import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch import Tensor
from typing import Union

tdict = dict[str, Tensor]


class Const:
    MODEL_NAME = "bert-base-uncased"
    MAX_TOKEN_LEN = 256
    SPECIAL_TOKENS = [
        "<head>",
        "</head>",
        "<tail>",
        "</tail>",
        "<url>",
        "<user>",
        "<date>",
        "<number>",
        "<money>",
        "<email>",
        "<percent>",
        "<phone>",
        "<time>",
        "<hashtag>",
        "</hashtag>",
    ]
    NORMALIZE = [
        "url",
        "email",
        "percent",
        "money",
        "phone",
        "user",
        "time",
        "url",
        "date",
        "number",
    ]


class Label:
    def __init__(self, name: str):
        """
        Class used to create labels based on task.

        Parameters
        ----------
        name : str
            Name of task, either GER or REL.
        """
        self.name = name
        assert self.name in {"GER", "REL"}, "Type must be either GER or REL"

        if self.name == "GER":
            self.labels: dict[str, int] = {
                "O": 0,
                "B-location": 1,
                "I-location": 2,
                "L-location": 3,
                "U-location": 4,
            }
        elif self.name == "REL":
            self.labels: dict[str, int] = {
                "contains": 0,
                "none": 1,
            }

        self.idx: dict[int, str] = {v: k for k, v in self.labels.items()}
        self.count: int = len(self.labels)


def encode_labels(
    tokens: list[str],
    labels: list[int],
    tokenizer,
    max_token_len: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """
    Encode list of tokens and labels into subwords.

    Input should be list of string tokens and integer labels of the same length,
    tokenizer should be pre-trained and have `add_prefix_space=True`.
    Output gives the encoding with token `input_ids` and `attention_mask`,
    `labels_encoded` gives the original label sequence but with -100 added where
    words have been split into sub words, or there is padding.

    Parameters
    ----------
    tokens : list[str]
        Pre-tokenized words in sequence
    labels : list[int]
        Label IDs for each token in sequence
    tokenizer : transformers.PretrainedTokenizer
        HuggingFace tokenizer
    max_token_len : int
        Cutoff length for number of tokens

    Returns
    -------
    tuple[dict, np.ndarray]:
        Encoded subwords and labels

    Example
    -------

    >>> from transformers import AutoTokenizer
    >>> tokens = ['Testing', 'this', 'for', 'doctest', '.']
    >>> labels = [1, 0, 0, 0, 0]
    >>> tokenizer = AutoTokenizer.from_pretrained(
    ...     'distilbert-base-cased', add_prefix_space=True
    ... )
    >>> encoding, labels_encoded = encode_labels(tokens, labels, tokenizer, 32)

    >>> len(encoding['input_ids'].flatten()) == len(labels_encoded)
    True
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        max_length=max_token_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    offset = np.array(encoding.offset_mapping[0])
    doc_enc_labels = np.ones(max_token_len, dtype=int) * -100  # type: ignore
    offsets_array = (offset[:, 0] == 0) & (offset[:, 1] != 0)
    if sum(offsets_array) < len(labels):
        doc_enc_labels[offsets_array] = labels[: sum(offsets_array)]
    else:
        doc_enc_labels[offsets_array] = labels

    encoded_labels = torch.LongTensor(doc_enc_labels)
    return encoding, encoded_labels


def combine_subwords(tokens: list[str], tags: list[int]) -> tuple[list[str], list[str]]:
    """
    Combines subwords and their tags into normal words with special chars removed.

    Parameters
    ----------
    tokens : list[str]
        Subword tokens.
    tags : list[int]
        Token tags of same length.

    Returns
    -------
    tuple[list[str], list[str]]:
        Combined tokens and tags.

    Example
    -------

    >>> tokens = ['Very', '##long', '##word', 'for', 'doct', '##est', '.']
    >>> tags = [4, -100, -100, 0, 4, -100, 0]
    >>> tokens, tags = combine_subwords(tokens, tags)

    >>> tokens
    ['Verylongword', 'for', 'doctest', '.']
    >>> len(tags) == len(tokens)
    True
    """
    idx = [
        idx
        for idx, token in enumerate(tokens)
        if token not in ["[CLS]", "[PAD]", "[SEP]"]
    ]

    tokens = [tokens[i] for i in idx]
    tags = [tags[i] for i in idx]

    for idx, _ in enumerate(tokens):
        idx += 1
        if tokens[-idx + 1].startswith("##"):
            tokens[-idx] = tokens[-idx] + tokens[-idx + 1][2:]
    subwords = [i for i, _ in enumerate(tokens) if not tokens[i].startswith("##")]

    tags = [tags[i] for i in subwords]
    tokens = [tokens[i] for i in subwords]
    tags_str: list[str] = [Label("GER").idx[i] for i in tags]
    return tokens, tags_str


def combine_biluo(tokens: list[str], tags: list[str]) -> tuple[list[str], list[str]]:
    """
    Combines multi-token BILUO tags into single entities.

    Parameters
    ----------
    tokens : list[str]
        Input tokenized string.
    tags : list[str]
        Tags corresponding with each token with BILUO format.

    Returns
    -------
    tuple[list[str], list[str]]:
        Tokens and tags with BILUO removed.

    Example:

    >>> tokens = ['New', 'York', 'City', 'is', 'big', '.']
    >>> tags = ['B-PLACE', 'I-PLACE', 'L-City', 'O', 'O', 'O']
    >>> tokens, tags = combine_biluo(tokens, tags)

    >>> tokens
    ['New York City', 'is', 'big', '.']
    >>> tags
    ['PLACE', 'O', 'O', 'O']
    """
    tokens_biluo = tokens.copy()
    tags_biluo = tags.copy()

    for idx, tag in enumerate(tags_biluo):
        if idx + 1 < len(tags_biluo) and tag[0] == "B":
            i = 1
            while tags_biluo[idx + i][0] not in ["B", "O"]:
                tokens_biluo[idx] = tokens_biluo[idx] + " " + tokens_biluo[idx + i]
                i += 1
                if idx + i == len(tokens_biluo):
                    break
    zipped = [
        (token, tag)
        for (token, tag) in zip(tokens_biluo, tags_biluo)
        if tag[0] not in ["I", "L"]
    ]
    tokens_biluo, tags_biluo = zip(*zipped)
    tags_biluo = [tag[2:] if tag != "O" else tag for tag in tags_biluo]
    return list(tokens_biluo), tags_biluo


def ents_to_relations(tokens: list[str], tags: list[str]) -> Union[list[str], None]:
    """
    Convert list of combined entities and other tokens with tags into semeval format.

    :param tokens list[str]: [TODO:description]
    :param tags list[str]: [TODO:description]
    :rtype Union[list[str], None]: [TODO:description]

    Example:

    >>> tokens = ['New York', 'is', 'in', 'New York', ',', 'America']
    >>> tags = ['PLACE', 'O', 'O', 'PLACE', 'O', 'PLACE']

    >>> ents_to_relations(tokens, tags)[0]
    '<head> New York </head> is in <tail> New York </tail> , America'
    >>> ents_to_relations(tokens, tags)[1]
    '<head> New York </head> is in New York , <tail> America </tail>'
    """

    # check if at least two entities
    if Counter(tags)["O"] > len(tags) - 2:
        return

    loc_idxs = [idx for idx, tag in enumerate(tags) if tag != "O"]
    sequence_list = []
    for i in loc_idxs:
        for j in loc_idxs:
            if i != j:
                tokens_copy = tokens.copy()
                tokens_copy[i] = "<head> " + tokens_copy[i] + " </head>"
                tokens_copy[j] = "<tail> " + tokens_copy[j] + " </tail>"
                sequence_list.append(" ".join(tokens_copy))
    return sequence_list


def convert_examples_to_features(
    item: Union[pd.Series, str],
    max_seq_len: int,
    tokenizer,
    labels: bool = True,
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    pad_token: int = 0,
    add_sep_token: bool = False,
    mask_padding_with_zero: bool = True,
):
    label_id = None
    if labels:
        tokens_a = tokenizer.tokenize(
            item["sentence"],
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_id = int(item["relation"])
    else:
        tokens_a = tokenizer.tokenize(
            item,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    # if tokenization shortens the string
    if any(x not in item for x in ["<head>", "</head>", "<tail>", "</tail>"]):
        return None

    e11_p = tokens_a.index("<head>")  # the start position of entity1
    e12_p = tokens_a.index("</head>")  # the end position of entity1
    e21_p = tokens_a.index("<tail>")  # the start position of entity2
    e22_p = tokens_a.index("</tail>")  # the end position of entity2

    # Replace the token
    tokens_a[e11_p] = "$"
    tokens_a[e12_p] = "$"
    tokens_a[e21_p] = "#"
    tokens_a[e22_p] = "#"

    # Add 1 because of the [CLS] token
    e11_p += 1
    e12_p += 1
    e21_p += 1
    e22_p += 1

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 2 if add_sep_token else 1
    if len(tokens_a) > max_seq_len - special_tokens_count:
        tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

    tokens = tokens_a
    if add_sep_token:
        tokens += [sep_token]

    tokens = [cls_token] + tokens
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + (
        [0 if mask_padding_with_zero else 1] * padding_length
    )

    # e1 mask, e2 mask
    e1_mask = [0] * len(attention_mask)
    e2_mask = [0] * len(attention_mask)

    for i in range(e11_p, e12_p + 1):
        e1_mask[i] = 1
    for i in range(e21_p, e22_p + 1):
        e2_mask[i] = 1

    assert (
        len(input_ids) == max_seq_len
    ), f"Error with input length {len(input_ids)} vs {max_seq_len}"
    assert (
        len(attention_mask) == max_seq_len
    ), f"Error with attention mask length {len(attention_mask)} vs {max_seq_len}"

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(label_id, dtype=torch.long) if labels else None,
        "e1_mask": torch.tensor(e1_mask, dtype=torch.long),
        "e2_mask": torch.tensor(e2_mask, dtype=torch.long),
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
