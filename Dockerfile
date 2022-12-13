ARG BASE_DOCKER_IMAGE=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${BASE_DOCKER_IMAGE}

# https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y libopenblas-dev nasm autoconf libtool google-perftools
RUN  python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# runtime knobs
ENV ONEFLOW_MLIR_CSE 1
ENV ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION 1
ENV ONEFLOW_MLIR_ENABLE_ROUND_TRIP 1
ENV ONEFLOW_MLIR_FUSE_FORWARD_OPS 1
ENV ONEFLOW_MLIR_GROUP_MATMUL 1
ENV ONEFLOW_MLIR_PREFER_NHWC 1
ENV ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS 1
ENV ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR 1
ENV ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL 1
ENV ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL 1
ENV ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL 1
ENV ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION 1
ENV ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION 1

ARG BUILD_FROM_SOURCE=1

# install oneflow with pip
ARG ONEFLOW_PIP_INDEX=https://staging.oneflow.info/branch/master/cu112
ARG ONEFLOW_PACKAGE_NAME=oneflow
RUN if [ "$BUILD_FROM_SOURCE" == "0" ] ; then \
    python3 -m pip install -f ${ONEFLOW_PIP_INDEX} --pre ${ONEFLOW_PACKAGE_NAME} ; \
    fi;

# build oneflow from source
# branch master
ARG ONEFLOW_COMMIT_ID=44e3d04982f379a4b513114c12fbf6094f80d901
ARG CUDAARCHS
RUN if [ "$BUILD_FROM_SOURCE" == "1" ] ; then \
    git clone https://github.com/Oneflow-Inc/oneflow /oneflow \
    && cd /oneflow \
    && git checkout ${ONEFLOW_COMMIT_ID} \
    && python3 -m pip install -r /oneflow/dev-requirements.txt \
    && mkdir /oneflow/build \
    && cd /oneflow/build \
    && cmake -DWITH_MLIR=YES -DWITH_MLIR_CUDA_CODEGEN=YES -DUSE_SYSTEM_NCCL=ON .. -C ../cmake/caches/cn/cuda.cmake \
    && make -j`nproc` ; \
    fi;
ENV PYTHONPATH /oneflow/python

# install diffusers
# branch oneflow-fork
ARG DIFFUSERS_COMMIT_ID=8c6abfe831e9e99f8f44ff657a1d41216c191fd8
RUN git clone https://github.com/Oneflow-Inc/diffusers /diffusers && cd /diffusers && git checkout ${DIFFUSERS_COMMIT_ID}
RUN cd /diffusers && python3 -m pip install -e .[oneflow]

# install transformers
# branch oneflow-fork
ARG TRANSFORMERS_COMMIT_ID=4e235cd3282d0afa920b3759ac959e35a94fc3ce
RUN git clone https://github.com/Oneflow-Inc/transformers /transformers && cd /transformers && git checkout ${TRANSFORMERS_COMMIT_ID}
RUN cd /transformers && python3 -m pip install -e .

ADD scripts /scripts
WORKDIR /scripts

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

