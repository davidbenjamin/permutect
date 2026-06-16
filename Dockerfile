# Base image provides Python 3.12 and CUDA 13.0
# nvidiaDriverVersion must be at least 570.86.15 on linux to support CUDA 13.0:
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
# Google cloud makes driver version recommendations for GCE VMs here:
# https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /app

# extra utilities for WDL tasks -- command line tools, not python packages
ARG bcftoolsVer="1.21"

# install system dependencies (consolidated into one layer)
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    bzip2 \
    autoconf \
    automake \
    make \
    gcc \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    libssl-dev \
    libgsl0-dev \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# build and install bcftools, then clean up source
RUN wget -q https://github.com/samtools/bcftools/releases/download/${bcftoolsVer}/bcftools-${bcftoolsVer}.tar.bz2 \
    && tar -xjf bcftools-${bcftoolsVer}.tar.bz2 \
    && rm bcftools-${bcftoolsVer}.tar.bz2 \
    && cd bcftools-${bcftoolsVer} \
    && make \
    && make install \
    && cd / \
    && rm -rf bcftools-${bcftoolsVer}

# install GATK launcher and jar
RUN wget -q -O /root/gatk.jar https://storage.googleapis.com/broad-dsp-david-benjamin/gatk-builds/active-regions.jar
RUN wget -q -O /bin/gatk https://raw.githubusercontent.com/broadinstitute/gatk/refs/heads/master/gatk \
    && chmod +x /bin/gatk
ENV GATK_LOCAL_JAR=/root/gatk.jar

# install permutect: copy pyproject.toml first for better layer caching
COPY pyproject.toml /app/
COPY permutect/ /app/permutect/

RUN pip install --break-system-packages --no-cache-dir .

CMD ["/bin/sh"]
