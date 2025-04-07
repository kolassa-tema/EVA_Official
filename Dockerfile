ARG CUDA_VERSION=12.1.0
ARG OS_VERSION=22.04


# Pull micromamba
FROM mambaorg/micromamba:1.5.10 AS micromamba

# Pull cuda
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}

# Set up micromamba
USER root
ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

CMD ["/bin/bash"]



# Cuda Environment Variables
ARG CUDA_ARCHITECTURES=86;89;90
ARG TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0

ENV CUDA_HOME="/usr/local/cuda"
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
# Set environment variables from build arguments
ENV CUDA_VERSION=${CUDA_VERSION}
ENV CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}



ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
USER root

# Install common tools
RUN apt update --fix-missing && apt install -y wget gnupg2 git cmake curl unzip screen
RUN apt install -y python3 python3-pip python3-venv \
libglew-dev libgl1-mesa-dev libglib2.0-0 libopencv-dev protobuf-compiler libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev

WORKDIR /workspace

# Note: Comment out nonexistent files
COPY ./requirements.txt* /workspace/requirements.txt
COPY ./environment*.y*ml /workspace/environment.yaml
COPY ./third_party /workspace/third_party



# Install environment with micromamba
RUN --mount=type=cache,target=/opt/conda/pkgs --mount=type=cache,target=/root/.cache/pip micromamba install -y -n base -f ./environment.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1