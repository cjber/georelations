FROM cjber/cuda as ENV

RUN pacman -S python-pip --noconfirm \
    && pip install poetry

WORKDIR /georelations

COPY pyproject.toml pyproject.toml
RUN poetry install

COPY . .

RUN echo 'export PROJECT_ROOT="/georelations"' > .env

ENTRYPOINT ["poetry", "run", "python", "-m", "src.run"]
