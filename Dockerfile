# syntax=docker/dockerfile:1.4

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel AS tslib

WORKDIR /workspace

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV PYTHONPATH=/workspace/Time-Series-Library:$PYTHONPATH

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# mamba-ssm (cxx11abiFALSE) （Time-Series-Library/models/Mamba.py）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# uni2ts (--no-deps)（Time-Series-Library/models/Moirai.py）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uni2ts --no-deps

COPY . .

CMD ["bash"]