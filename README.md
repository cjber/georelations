# Geographic Relation Classification

<p align="center">
<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
</p>
<!-- <p align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge"></a>
<a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue?style=for-the-badge"></a>
<a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow?style=for-the-badge"></a>
<a href="https://dvc.org/"><img alt="Conf: hydra" src="https://img.shields.io/badge/data-dvc-9cf?style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge"></a>
<a href="https://github.com/lucmos/nn-template?style=for-the-badge"><img alt="Template: nn-template" src="https://shields.io/badge/-nn--template-emerald?style=for-the-badge&logo=github&labelColor=gray"></a>
</p> -->

Fine-tuned DistilBERT entity recognition and classification models for
![Relation
Extraction](https://paperswithcode.com/task/relation-extraction),
focussing purely on geographic entities.

Entity recognition uses a basic token classification head, trained using
custom data focussing on geographic entities. Relation extraction
mirrors the `R-BERT` architecture: [Enriching Pre-trained Language Model
and Entity Information for Relation
Classification](https://arxiv.org/abs/1905.08284), taking code from the
[*Unofficial* PyTorch
implementation](https://github.com/monologg/R-BERT).

## Training

#### Build poetry environment (inside venv)

``` commandline
make env
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
