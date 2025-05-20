FROM ubuntu:20.04
LABEL author="Adam Ewing <adam.ewing@gmail.com>"
LABEL maintainer="Julian Gascoyne <jmgascoyne@mdanderson.com>"
LABEL description="This image contains the tools required for bamsurgeon"
LABEL version="1.4.1"
LABEL url="https://github.com/adamewing/bamsurgeon.git"
LABEL usage="docker run -ti --rm bamsurgeon:1.4.1"
ENV PATH=$PATH:./bin
ARG DEBIAN_FRONTEND=noninteractive
# Install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    git \
    wget \
    build-essential \
    libz-dev \
    libglib2.0-dev \
    libbz2-dev \
    liblzma-dev \
    libhts-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    zlib1g-dev \
    default-jre \
    autoconf \
    checkinstall \
    ca-certificates \
    && apt-get clean
# Install Python 3.6.8 from source
RUN wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz \
    && tar -xvf Python-3.6.8.tgz \
    && cd Python-3.6.8 \
    && ./configure --enable-optimizations \
    && make \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.6.8*
# Install pip for Python 3.6.8
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py \
    && /usr/local/bin/python3.6 get-pip.py \
    && rm get-pip.py
# Set Python 3.6 as the default python3
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.6 1
# Install Samtools 1.9 from source
RUN wget https://github.com/samtools/samtools/releases/download/1.9/samtools-1.9.tar.bz2 \
    && tar -xvjf samtools-1.9.tar.bz2 \
    && cd samtools-1.9 \
    && make \
    && make install \
    && cd .. \
    && rm -rf samtools-1.9*
# Download and install BWA 0.7.12
RUN wget https://sourceforge.net/projects/bio-bwa/files/bwa-0.7.12.tar.bz2/download -O bwa-0.7.12.tar.bz2 \
    && tar -xvjf bwa-0.7.12.tar.bz2 \
    && cd bwa-0.7.12 \
    && make \
    && mv bwa /usr/local/bin/ \
    && cd .. \
    && rm -rf bwa-0.7.12 bwa-0.7.12.tar.bz2
# Retrieve and install velvet:1.2.10 from GitHub
RUN wget https://github.com/dzerbino/velvet/archive/refs/tags/v1.2.10.tar.gz \
    && tar -xvzf v1.2.10.tar.gz \
    && make -C velvet-1.2.10 \
    && cp velvet-1.2.10/velvetg /usr/local/bin/ \
    && cp velvet-1.2.10/velveth /usr/local/bin/
# Retrieve and install exonerate from GitHub
RUN git clone https://github.com/adamewing/exonerate.git --branch v2.4.0 \
    && cd exonerate \
    && autoreconf -fi \
    && ./configure && make && make install \
    && cd .. \
    && rm -rf exonerate
# Retrieve and install Picard from GitHub
RUN wget https://github.com/broadinstitute/picard/releases/download/2.18.9/picard.jar \
    && chmod +x picard.jar \
    && cp picard.jar /picard.jar && mv picard.jar /usr/local/bin/ \
    && export BAMSURGEON_PICARD_JAR=/usr/local/bin/picard.jar
# Install Pysam
RUN pip install pysam==0.15.2
# Retrieve and install bamsurgeon from GitHub
RUN git clone https://github.com/adamewing/bamsurgeon.git --branch 1.4.1 \
    && export PATH=$PATH:/usr/local/bin/ && cd bamsurgeon
# Set default command
CMD ["/bin/bash"]
