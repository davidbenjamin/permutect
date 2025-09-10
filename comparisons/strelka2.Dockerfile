FROM ubuntu:16.04

LABEL maintainer="Julian Gascoyne <jmgascoyne@mdanderson.org>"
LABEL description="This image contains the Strelka and Manta tools"
LABEL usage="docker run -ti --rm strelka:2.9.10"

# Build-time arguments for versioning
ARG PYTHON_VERSION=2.7.18
ARG PYTHON_MAJOR_MINOR=2.7

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MANTA_VERSION=1.6.0
ENV STRELKA_VERSION=2.9.10


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    gnupg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    zlib1g-dev \
    libboost-all-dev \
    ca-certificates \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python from source
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python${PYTHON_MAJOR_MINOR} /usr/local/bin/python && \
    ln -s /usr/local/bin/pip${PYTHON_MAJOR_MINOR} /usr/local/bin/pip || true

RUN wget https://bootstrap.pypa.io/pip/${PYTHON_MAJOR_MINOR}/get-pip.py -O get-pip.py && \
    /usr/local/bin/python${PYTHON_MAJOR_MINOR} get-pip.py && \
    rm get-pip.py && \
    update-alternatives --install /usr/local/bin/python python /usr/local/bin/python${PYTHON_MAJOR_MINOR} 1

# Clone and build Manta
WORKDIR /opt
RUN wget https://github.com/Illumina/manta/releases/download/v${MANTA_VERSION}/manta-${MANTA_VERSION}.release_src.tar.bz2 && \
    tar -xjf manta-${MANTA_VERSION}.release_src.tar.bz2 && \
    mkdir /opt/manta-build && cd /opt/manta-build && \
    /opt/manta-${MANTA_VERSION}.release_src/configure && \
    make -j$(nproc) install

# Clone and build Strelka
WORKDIR /opt
RUN wget https://github.com/Illumina/strelka/releases/download/v${STRELKA_VERSION}/strelka-${STRELKA_VERSION}.release_src.tar.bz2 && \
    tar -xjf strelka-${STRELKA_VERSION}.release_src.tar.bz2 && \
    mkdir /opt/strelka-build && \
    cd /opt/strelka-build && \
    /opt/strelka-${STRELKA_VERSION}.release_src/configure && \
    make -j$(nproc) install

# Mount the binary locations on PATH and set default run command
WORKDIR /data
CMD ["/bin/bash"]
