.PHONY: docs

PYTHON = python3
# generated from import random;random.randint(1000) x 5
SEEDS = 487 726 231 879 323

env:
	poetry install

pytest:
## pytest: run pytest doctest and unit tests
	poetry run python -m pytest \
		&& poetry run python -m coverage run -m pytest --doctest-modules src/common

clean:
## clean: remove all experiments and cache files
	rm -rf .pytest_cache \
	    && find . -type d -iname '__pycache__' -exec rm -rf {} + \
	    && rm -rf ckpts/*

docs:
## docs: build documentation automatically
	poetry run python -m pdoc --html --force --output-dir docs src

lint:
## lint: lint check all source files using black and flake8
	poetry run python -m black src --check --diff \
	    && poetry run flake8 --ignore E203,E501,W503,F841,F401 src

run:
## run: Train ger and rel model over 5 fixed seeds.
	poetry run ${PYTHON} -m src.run --model ger --seed ${SEEDS} --batch_size 4 \
	&& poetry run ${PYTHON} -m src.run --model rel --seed ${SEEDS}

dev:
## dev: Test ger and rel models train.
	poetry run ${PYTHON} -m src.run --model ger --fast_dev_run True \
	&& poetry run ${PYTHON} -m src.run --model rel --fast_dev_run True

help:
## help: This helpful list of commands
	@echo "Usage: "
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/-/'
