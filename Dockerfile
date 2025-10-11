# Base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUTF8=1


# Install common utilities, Git, Python 3.11, pip, and GitHub CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    nvidia-utils-525 \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Make Git ignore permission bit changes and normalise line endings
RUN git config --global core.fileMode false && \
    git config --global core.autocrlf input


# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install pipreqs
RUN pip install --no-cache-dir pipreqs

# Set working directory
WORKDIR /workspaces

# Copy source code into image for dependency detection
COPY . /workspaces

# Generate requirements into a safe location, install CuPy wheel first, then the rest
RUN pipreqs . --force --savepath /tmp/requirements.txt && \
    sed -i 's/^cupy.*$/cupy-cuda12x==13.6.0/' /tmp/requirements.txt && \
    pip install --no-cache-dir cupy-cuda12x==13.6.0 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir Pillow && \
    pip install --no-cache-dir matplotlib && \
    pip install --no-cache-dir pandas

CMD [ "bash" ]
