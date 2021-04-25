from collections import Counter
import copy
import json
import os
import string
from typing import Optional, Union

import dotenv
import numpy as np
import pandas as pd


class Config:
    """Label params"""

    GER_LABELS: dict[str, int] = {
        "B-PLACE": 0,
        "I-PLACE": 1,
        "L-PLACE": 2,
        "U-PLACE": 3,
        "O": 4,
    }
    GER_IDX: dict[int, str] = {v: k for k, v in GER_LABELS.items()}
    GER_NUM: int = len(GER_LABELS)

    REL_LABELS: dict[str, int] = {"NONE": 0, "NTPP": 1, "DC": 2}
    REL_IDX: dict[int, str] = {v: k for k, v in REL_LABELS.items()}
    REL_NUM: int = len(REL_LABELS)

    """Model params"""

    MAX_TOKEN_LEN: int = 128
    MODEL_NAME: str = "roberta-base"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        e1_mask,
        e2_mask,
        label_id: Union[int, None],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def encode_labels(
    tokens: list[str], labels: list[int], tokenizer
) -> tuple[dict, np.ndarray]:
    """
    Encode list of tokens and labels into subwords.

    Input should be list of string tokens and integer labels of the same length,
    tokenizer should be pre-trained and have `add_prefix_space=True`.
    Output gives the encoding with token `input_ids` and `attention_mask`,
    `labels_encoded` gives the original label sequence but with -100 added where
    words have been split into sub words, or there is padding.

    :param tokens list[str]: Pre-tokenized words in sequence
    :param labels list[int]: Label IDs for each token in sequence
    :param tokenizer PretrainedTokenizer: HuggingFace tokenizer
    :rtype tuple[BatchEncoding, np.ndarray]: Encoded subwords and labels

    Example:

    >>> from transformers import AutoTokenizer
    >>> tokens = ['Testing', 'this', 'for', 'doctest', '.']
    >>> labels = [1, 0, 0, 0, 0]
    >>> tokenizer = AutoTokenizer.from_pretrained(
    ...     Config.MODEL_NAME, add_prefix_space=True
    ... )
    >>> encoding, labels_encoded = encode_labels(tokens, labels, tokenizer)

    >>> len(encoding['input_ids'].flatten()) == len(labels_encoded)
    True
    """

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_attention_mask=True,
        max_length=Config.MAX_TOKEN_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    subwords = np.array(
        tokenizer.tokenize(
            tokens,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=Config.MAX_TOKEN_LEN,
            padding="max_length",
            truncation=True,
        )
    )

    word_idx = [
        idx
        for idx, word in enumerate(subwords)
        if word[0] == "\u0120"  # all non subwords start with this
    ]
    labels_encoded = np.ones(len(subwords)) * -100
    # if truncated remove extra labels
    if len(labels) > (x := len(word_idx)):
        labels = labels[:x]
    labels_encoded[word_idx] = labels

    return encoding, labels_encoded


def combine_subwords(tokens: list[str], tags: list[int]) -> tuple[list[str], list[str]]:
    """
    Combines subwords and their tags into normal words with special chars removed.

    NOTE: This currently removes all punctuation!

    Example:

    >>> tokens = ['ĠVery', 'long', 'word', 'Ġfor', 'Ġdoct', 'est', 'Ġ.']
    >>> tags = [1, -100, -100, 0, 1, -100, 0]
    >>> tokens, tags = combine_subwords(tokens, tags)

    >>> tokens
    ['Verylongword', 'for', 'doctest', '.']
    >>> len(tags) == len(tokens)
    True

    :param tokens list[str]: List of subword tokens.
    :param tags list[int]: List of tags of same length.
    :rtype tuple[list[str], list[str]]: Combined tokens and tags
    """
    idx = [
        idx for idx, token in enumerate(tokens) if token not in ["<pad>", "<s>", "</s>"]
    ]
    tokens = [tokens[i] for i in idx]
    tags = [tags[i] for i in idx]

    for idx, _ in enumerate(tokens):
        idx += 1
        if (
            tokens[-idx + 1][0] != "\u0120"
            and tokens[-idx + 1] not in string.punctuation
        ):
            tokens[-idx] = tokens[-idx] + tokens[-idx + 1]
    idx = [idx for idx, token in enumerate(tokens) if token[0] == "\u0120"]

    tokens = [tokens[i][1:] for i in idx]
    tags_str: list[str] = [Config.GER_IDX[tags[i]] for i in idx]
    return tokens, tags_str


def combine_biluo(tokens: list[str], tags: list[str]) -> tuple[list[str], list[str]]:
    """
    Combines multi-token BILUO tags into single entities.

    :param tokens list[str]: [TODO:description]
    :param tags list[str]: [TODO:description]
    :rtype tuple[list[str], list[str]]: [TODO:description]

    Example:

    >>> tokens = ['New', 'York', 'is', 'big', '.']
    >>> tags = ['B-PLACE', 'L-PLACE', 'O', 'O', 'O']
    >>> tokens, tags = combine_biluo(tokens, tags)

    >>> tokens
    ['New York', 'is', 'big', '.']
    >>> tags
    ['PLACE', 'O', 'O', 'O']
    """

    tokens_biluo = tokens.copy()
    tags_biluo = tags.copy()

    for idx, tag in enumerate(tags_biluo):
        if idx + 1 < len(tags_biluo) and tag[0] == "B":
            i = 1
            while tags_biluo[idx + i][0] not in ["B", "O", "U"]:
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
    '<e1> New York </e1> is in <e2> New York </e2> , America'
    """

    # check if at least two entities
    if Counter(tags)["O"] <= len(tags) - 2:
        first_entity_idx = next(idx for idx, tag in enumerate(tags) if tag != "O")
        other_entity_idx = [
            idx
            for idx, tag in enumerate(tags)
            if (idx != first_entity_idx) & (tag != "O")
        ]

        tokens[first_entity_idx] = "<e1> " + tokens[first_entity_idx] + " </e1>"
        sequence_list = []
        for idx in other_entity_idx:
            tokens_copy = tokens.copy()
            tokens_copy[idx] = "<e2> " + tokens_copy[idx] + " </e2>"
            sequence_list.append(" ".join(tokens_copy))
        return sequence_list


def convert_examples_to_features(
    item: pd.Series,
    max_seq_len: int,
    tokenizer,
    labels: bool = True,
    cls_token: str = "[CLS]",
    cls_token_segment_id: int = 0,
    sep_token: str = "[SEP]",
    pad_token: int = 0,
    pad_token_segment_id: int = 0,
    sequence_a_segment_id: int = 0,
    add_sep_token: bool = False,
    mask_padding_with_zero: bool = True,
):
    label_id = None
    if labels:
        tokens_a = tokenizer.tokenize(item.text_a)
        label_id = int(item.label)
    else:
        tokens_a = tokenizer.tokenize(item)

    e11_p = tokens_a.index("<e1>")  # the start position of entity1
    e12_p = tokens_a.index("</e1>")  # the end position of entity1
    e21_p = tokens_a.index("<e2>")  # the start position of entity2
    e22_p = tokens_a.index("</e2>")  # the end position of entity2

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

    token_type_ids = [sequence_a_segment_id] * len(tokens)
    tokens = [cls_token] + tokens
    token_type_ids = [cls_token_segment_id] + token_type_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + (
        [0 if mask_padding_with_zero else 1] * padding_length
    )
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

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
    assert (
        len(token_type_ids) == max_seq_len
    ), f"Error with token type length {len(token_type_ids)} vs {max_seq_len}"

    return InputFeatures(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        label_id=label_id,
        e1_mask=e1_mask,
        e2_mask=e2_mask,
    )


def get_env(env_name: str) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        raise KeyError(f"{env_name} not defined")
    env_value: str = os.environ[env_name]
    if not env_value:
        raise ValueError(f"{env_name} has yet to be configured")
    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
