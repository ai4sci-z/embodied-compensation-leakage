ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG PIP_VERSION=26.2.1
ARG SETUPTOOLS_VERSION=84.0.0
ARG WHEEL_VERSION=0.48.0

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglfw3 \
        libglib2.0-0 \
        libhdf5-dev \
        libosmesa6 \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY environment/docker/constraints-lock.txt /tmp/constraints-lock.txt

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install \
        "pip==${PIP_VERSION}" \
        "setuptools==${SETUPTOOLS_VERSION}" \
        "wheel==${WHEEL_VERSION}" \
    && python -m pip install \
        --constraint /tmp/constraints-lock.txt \
        --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"

ARG ROBOMIMIC_COMMIT=ae5799f0fae05c4559ee1f9645b0f77eb5251929
ENV ROBOMIMIC_COMMIT=${ROBOMIMIC_COMMIT}

COPY environment/docker/constraints-cu128.txt /tmp/constraints-cu128.txt
COPY environment/docker/robomimic-0.5.0-dependencies.patch /tmp/robomimic-dependencies.patch

RUN git clone --filter=blob:none https://github.com/ARISE-Initiative/robomimic.git /opt/robomimic \
    && git -C /opt/robomimic checkout --detach "${ROBOMIMIC_COMMIT}" \
    && git -C /opt/robomimic apply --check /tmp/robomimic-dependencies.patch \
    && git -C /opt/robomimic apply /tmp/robomimic-dependencies.patch \
    && test "$(git -C /opt/robomimic rev-parse HEAD)" = "${ROBOMIMIC_COMMIT}"

WORKDIR /workspace/PORTAL-Trace

COPY pyproject.toml README.md ./
COPY portal_trace ./portal_trace
COPY third_party ./third_party

RUN sed -i \
        's|robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0|robomimic @ file:///opt/robomimic|' \
        pyproject.toml \
    && grep -Fq 'robomimic @ file:///opt/robomimic' pyproject.toml \
    && python -m pip install \
        --constraint /tmp/constraints-cu128.txt \
        --constraint /tmp/constraints-lock.txt \
        -e ".[dev]" \
    && python -m pip check

RUN existing_group="$(getent group "${GROUP_ID}" | cut -d: -f1)" \
    && if [ -n "${existing_group}" ] && [ "${existing_group}" != portal ]; then \
        groupmod --new-name portal "${existing_group}"; \
       elif [ -z "${existing_group}" ]; then \
        groupadd --gid "${GROUP_ID}" portal; \
       fi \
    && existing_user="$(getent passwd "${USER_ID}" | cut -d: -f1)" \
    && if [ -n "${existing_user}" ] && [ "${existing_user}" != portal ]; then \
        usermod --login portal --home /home/portal --move-home --gid "${GROUP_ID}" "${existing_user}"; \
       elif [ -z "${existing_user}" ]; then \
        useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash portal; \
       fi \
    && mkdir -p /data /results /cache \
    && chown -R portal:portal /workspace /data /results /cache

USER portal
CMD ["bash"]
