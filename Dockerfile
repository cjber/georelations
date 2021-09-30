FROM docker.io/cjber/cuda as ENV

RUN pacman -Syu python-pip --noconfirm \
    && pip install poetry

WORKDIR /georelations

COPY pyproject.toml pyproject.toml
RUN poetry install

COPY src src
COPY data data

ENTRYPOINT ["poetry", "run", "python", "-m", "src.run"]
