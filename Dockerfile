FROM cjber/arch as ENV
ENV PATH "/home/cjber/.local/bin:$PATH"

RUN yay --noconfirm \
    && yay -S python-pip --noconfirm

WORKDIR /home/cjber/georelations

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM env

COPY --chown=cjber . ./

RUN echo 'export PROJECT_ROOT="/home/cjber/georelations"' > .env

# configure nvidia container runtime
# https://github.com/NVIDIA/nvidia-container-runtime#environment-variables-oci-spec
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["nvidia-smi"]
# ENTRYPOINT ["python", "-m", "src.run"]
