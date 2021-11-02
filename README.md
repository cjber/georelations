# Geographic Relationship Extraction

<p align="center">
<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
</p>

A two stage ensemble for extracting geographic relationships between two
named geographic entities in text.

Fine-tuned DistilBERT entity recognition and classification models for
![Relation
Extraction](https://paperswithcode.com/task/relation-extraction),
focussing purely on geographic entities.

> **Liverpool** is in **Merseyside**.

> (Merseyside, `contains`, Liverpool)

1.  **Entity recognition** uses a basic token classification head,
    trained using custom data focussing on geographic entities.
2.  **Relation extraction** mirrors the `R-BERT` architecture:
    [Enriching Pre-trained Language Model and Entity Information for
    Relation Classification](https://arxiv.org/abs/1905.08284), taking
    code from the [*Unofficial* PyTorch
    implementation](https://github.com/monologg/R-BERT).

## Training

#### Build poetry environment (inside venv)

``` commandline
make env
```

#### Train Models

``` commandline
python -m src.run --model ger
python -m src.run --model rel
```

#### Run inference

``` commandline
python -n src.inf
```

## Docker

#### Build from Dockerfile

``` bash
docker build . -t cjber/georelations
```

#### Run with volume mapped on GPU

``` bash
docker run --rm --gpus all -v ${PWD}/ckpts:/georelations/ckpts -v ${PWD}/csv_logs:/georelations/csv_logs cjber/georelations
```
