# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:19.09-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
LABEL repository="vectorizer"

# Install additional programs
RUN apt update && \
    apt install -y build-essential \
                   htop \
                   git \
                   curl \
                   ca-certificates \
                   vim \
                   tmux && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN SHA=ToUcHMe which python3
RUN SHA=ToUcHMe python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install dependencies
RUN python3 -m pip install tqdm
RUN python3 -m pip install Pillow
RUN python3 -m pip install numpy
RUN python3 -m pip install flair==0.4.5
RUN python3 -m pip install allennlp==0.9.0
RUN python3 -m pip install transformers==2.5.1
RUN python3 -m pip install attrdict

# Install additional dependencies (for VS Code dev)
RUN python3 -m pip install autopep8
RUN conda install pylint

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME 
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
RUN mkdir /home/$USER_NAME/.vscode-server 
RUN mkdir /home/$USER_NAME/.vscode-server-insiders

# Change owner of home dir
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

# Set workdir when starting container
WORKDIR /vectorizer

# Add workdir to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/vectorizer"

CMD ["/bin/bash"]
