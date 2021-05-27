.PHONY: docs

PYTHON = python3

env:
	- pip install -r requirements.txt

train-%:
## train-{model}: train full model either ger or rel
	${PYTHON} -m src.run --config-name=${@:train-%=%}

dev-%:
## dev-{model}: dev run of training
	${PYTHON} -m src.run --config-name=${@:dev-%=%} \
	    train.pl_trainer.fast_dev_run=True

doctest:
## doctest: run doctests
	pytest --doctest-modules src/common/model_utils.py

clean:
## clean: remove all experiments and cache files
	rm -rf .pytest_cache \
	    && find . -type d -iname '__pycache__' -exec rm -rf {} + \
	    && rm -rf wandb/* \
	    && rm -rf experiments/* \
	    && rm -rf outputs/*

docs:
## docs: build documentation automatically
	pdoc --html --force --output-dir docs src

black:
	black src
## black: autolint all source files using black

help:
## help: This helpful list of commands
	@echo "Usage: \n"
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/-/'
