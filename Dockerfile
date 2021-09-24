FROM docker.io/cjber/cuda as ENV

RUN pacman -S python-pip --noconfirm \
    && pip install poetry

WORKDIR /georelations

COPY pyproject.toml pyproject.toml
RUN poetry install

COPY data src .python-version

ENTRYPOINT ["poetry", "run", "python", "-m", "src.run"]
