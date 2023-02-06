ARG BASE_DOCKER_IMAGE=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${BASE_DOCKER_IMAGE}

# https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y libopenblas-dev nasm autoconf libtool google-perftools
RUN  python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

ARG BUILD_FROM_SOURCE=1

# install oneflow with pip
ARG ONEFLOW_PIP_INDEX=https://staging.oneflow.info/branch/master/cu117
ARG ONEFLOW_PACKAGE_NAME=oneflow
RUN if [ "$BUILD_FROM_SOURCE" == "0" ] ; then \
    python3 -m pip install -f ${ONEFLOW_PIP_INDEX} --pre ${ONEFLOW_PACKAGE_NAME} ; \
    fi;

# build oneflow from source
# branch master
ARG ONEFLOW_GIT_URL=https://github.com/Oneflow-Inc/oneflow.git
ARG ONEFLOW_COMMIT_ID=d85ac4c410069d1ae18795c665ccfdb47004d8a7
ARG CUDAARCHS
RUN if [ "$BUILD_FROM_SOURCE" == "1" ] ; then \
    git clone ${ONEFLOW_GIT_URL} /oneflow \
    && cd /oneflow \
    && git checkout ${ONEFLOW_COMMIT_ID} \
    && python3 -m pip install -r /oneflow/dev-requirements.txt \
    && mkdir /oneflow/build \
    && cd /oneflow/build \
    && cmake -DWITH_MLIR=YES -DUSE_SYSTEM_NCCL=ON .. -C ../cmake/caches/cn/cuda.cmake \
    && make -j`nproc` ; \
    fi;
ENV PYTHONPATH /oneflow/python

RUN python3 -m pip install transformers==4.26

# install diffusers
# branch oneflow-fork
ARG DIFFUSERS_GIT_URL=https://github.com/Oneflow-Inc/diffusers.git
ARG DIFFUSERS_COMMIT_ID=2d7a6faa39b1efb2df0b057645c560bd2aae5f8a
RUN git clone ${DIFFUSERS_GIT_URL} /diffusers && cd /diffusers && git checkout ${DIFFUSERS_COMMIT_ID}
RUN cd /diffusers && python3 -m pip install -e .[oneflow]

ADD scripts /scripts
RUN python3 -m pip install -r /scripts/requirements.txt
WORKDIR /scripts

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
