.PHONY: docs

PYTHON = python3
# generated from import random;random.randint(1000) x 5
SEEDS = 487 726 231 879 323

env:
	- poetry install

train:
## train: train full model
	${PYTHON} -m src.run

dev:
## dev: dev run of training
	${PYTHON} -m src.run --fast_dev_run=True

pytest:
## pytest: run pytest doctest and unit tests
	poetry run python -m coverage run -m pytest --doctest-modules src/common

clean:
## clean: remove all experiments and cache files
	rm -rf .pytest_cache \
	    && find . -type d -iname '__pycache__' -exec rm -rf {} + \
	    && rm -rf ckpts/*

docs:
## docs: build documentation automatically
	rm -r docs \
	    && poetry run python -m pdoc --html --force --output-dir docs \
		src/pl_data \
		src/pl_metric \
		src/pl_modules \
		src/common

lint:
## lint: lint check all source files using black and flake8
	poetry run python -m black src --check --diff \
	    && poetry run flake8 --ignore E203,E501,W503,F841,F401 src

run:
## run: Train ger and rel model over 5 fixed seeds.
	${PYTHON} -m src.run --model ger --seed ${SEEDS} \
	&& ${PYTHON} -m src.run --model rel --seed ${SEEDS}

help:
## help: This helpful list of commands
	@echo "Usage: \n"
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/-/'
