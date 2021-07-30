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

pytest:
## pytest: run pytest doctest and unit tests
	pytest --doctest-modules src/common \
	    && pytest

clean:
## clean: remove all experiments and cache files
	rm -rf .pytest_cache \
	    && find . -type d -iname '__pycache__' -exec rm -rf {} + \
	    && rm -rf wandb/* \
	    && rm -rf experiments/* \
	    && rm -rf outputs/* \
	    && rm -rf ckpt/*

docs:
## docs: build documentation automatically
	rm -r docs \
	    && pdoc --html --force --output-dir docs src \
	    && mv docs/src/* docs/ \
	    && rm -r docs/src

lint:
## lint: lint check all source files using black and flake8
	# --max-complexity 7: may re-add
	black src --check --diff \
	    && flake8 --ignore E501,W503,F841,F401 src

help:
## help: This helpful list of commands
	@echo "Usage: \n"
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/-/'
