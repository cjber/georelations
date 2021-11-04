<div align="center">

# Geographic Relationship Extraction

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>

</div>

## Description

A two stage ensemble for extracting geographic relationships between two
named geographic entities in text. Uses fine-tuned DistilBERT entity
recognition and classification models for [Relationship
Extraction](https://paperswithcode.com/task/relation-extraction),
focussing purely on geographic entities (place names).

> **Liverpool** is in **Merseyside**.

> (Merseyside, `contains`, Liverpool)

1.  **Entity recognition** uses a basic token classification head,
    trained using
    [WNUT17](http://noisy-text.github.io/2017/emerging-rare-entities.html),
    using only `location` entities.
2.  **Relation extraction** mirrors the `R-BERT` architecture:
    [Enriching Pre-trained Language Model and Entity Information for
    Relation Classification](https://arxiv.org/abs/1905.08284), taking
    code from the [*Unofficial* PyTorch
    implementation](https://github.com/monologg/R-BERT).

## Project layout

``` bash
src
├── common
│   └── utils.py  # various utility functions and constants
│
├── datasets
│   ├── wnut_dataset.py  # torch dataset for wnut data for ger model
│   ├── rel_dataset.py  # torch dataset for nyt-h data for rel model
│   ├── jsonl_dataset.py  # torch dataset for reddit comments data for inference
│   └── datamodule.py  # lightning datamodule
│
├── metrics
│   └── seqeval_f1.py  # seqeval f1 metric for pytorch lightning (BILUO)
│
├── modules
│   ├── ger_model.py  # ger token classification model
│   ├── rel_model.py  # rel sequence classification model
│   └── ensemble.py  # ensemble wrapper used for inference pipeline
│
├── processing
│   ├── process_nyth.py  # process nyt-h data into csv form
│   └── reddit_api.py  # use pushshift api to download reddit comments
│
├── run.py  # training loop for models
└── inf.py  # use ensemble model with checkpoints for reddit comment inference
```

## How to run

### Build poetry environment

Install dependencies using [Poetry](https://python-poetry.org/):

``` bash
poetry install
```

#### Train entity recognition model using [WNUT17](https://huggingface.co/datasets/wnut_17):

- Run model training loop

  > **NOTE:** Dataset will automatically download

    ``` bash
    poetry run python -m src.run --model ger
    ```

#### Train relationship classification model using [NYT-H corpus](https://github.com/Spico197/NYT-H).

1.  Download the corpus from the [NYT-H
    GitHub](https://github.com/Spico197/NYT-H)

2.  From the [File Structure and Data
    Preparation](https://github.com/Spico197/NYT-H#file-structure-and-data-preparation)
    section prepare `train.json`

    ``` bash
    cat train_nonna.json na_train.json > train.json
    ```

3.  Process `train.json` and `test.json` into
    `data/distant_data/{relations.csv,relations_test.csv}`

    ``` bash
    poetry run python -n src.processing.process_nyth.py
    ```

4.  Run model training loop

    ``` bash
    poetry run python -m src.run --model rel
    ```

#### Run inference

1.  Get corpus of Reddit comments from UK place subreddits using the
    [Pushshit API](https://github.com/pushshift/api)

    > **NOTE:** Using the Pushshift API for this volume takes a very
    > long time. It is recommended that only a single subreddit is used
    > (edit the python script)

    ``` bash
    poetry run python -m src.processing.reddit_api.py
    ```

2.  Run inference using model checkpoints

    ``` bash
    python -m src.inf
    ```

## Docker

> **NOTE:** Run as an alternative to locally installing poetry

#### Build from Dockerfile

``` bash
docker build . -t cjber/georelations
```

#### Run with volume mapped on GPU

``` bash
docker run --rm --gpus all -v ${PWD}/ckpts:/georelations/ckpts -v ${PWD}/csv_logs:/georelations/csv_logs cjber/georelations
```
