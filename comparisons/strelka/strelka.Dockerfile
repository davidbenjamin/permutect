# copied from https://github.com/cancerit/strelka2-manta

FROM ubuntu:16.04 as builder

USER root

ENV VER_MANTA="1.6.0"
ENV VER_STRELKA2="2.9.10"

RUN apt-get -yq update
# no benefit of combined in builder stage
RUN apt-get install -yq --no-install-recommends locales
RUN apt-get install -yq --no-install-recommends ca-certificates
RUN apt-get install -yq --no-install-recommends gcc
RUN apt-get install -yq --no-install-recommends g++
RUN apt-get install -yq --no-install-recommends python
RUN apt-get install -yq --no-install-recommends bzip2
RUN apt-get install -yq --no-install-recommends make
RUN apt-get install -yq --no-install-recommends zlib1g-dev
RUN apt-get install -yq --no-install-recommends cmake
RUN apt-get install -yq --no-install-recommends libboost-dev
RUN apt-get install -yq --no-install-recommends curl

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV OPT /opt/wtsi-cgp
ENV PERL5LIB $OPT/lib/perl5
ENV LD_LIBRARY_PATH $OPT/lib
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

# download and install manta
WORKDIR /tmp
RUN curl -sSL --retry 10 https://github.com/Illumina/manta/releases/download/v${VER_MANTA}/manta-${VER_MANTA}.release_src.tar.bz2 > manta.tar.bz2
RUN mkdir -p manta
RUN tar --strip-components 1 -C manta -jxf manta.tar.bz2
WORKDIR /tmp/manta/build
RUN ../configure --jobs=4 --prefix=$OPT/manta
RUN make -j4 install

# download and install strelka2
WORKDIR /tmp
RUN curl -sSL --retry 10 https://github.com/Illumina/strelka/releases/download/v${VER_STRELKA2}/strelka-${VER_STRELKA2}.release_src.tar.bz2 > strelka.tar.bz2
RUN mkdir -p strelka
RUN tar --strip-components 1 -C strelka -jxf strelka.tar.bz2
WORKDIR /tmp/strelka/build
RUN ../configure --jobs=4 --prefix=$OPT/strelka
RUN make -j4 install

FROM ubuntu:16.04

LABEL maintainer="cgphelp@sanger.ac.uk" \
      uk.ac.sanger.cgp="Cancer, Ageing and Somatic Mutation, Wellcome Trust Sanger Institute" \
      version="1.0.0" \
      description="Strelka2 docker"

RUN apt-get -yq update \
&& apt-get install -yq --no-install-recommends \
locales \
python \
bzip2 \
zlib1g \
curl \
unattended-upgrades && \
unattended-upgrade -d -v && \
apt-get remove -yq unattended-upgrades && \
apt-get autoremove -yq

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV OPT /opt/wtsi-cgp
ENV PATH $OPT/manta/bin:$PATH
ENV PATH $OPT/strelka/bin:$PATH
ENV PERL5LIB $OPT/lib/perl5
ENV LD_LIBRARY_PATH $OPT/lib
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

RUN mkdir -p $OPT
COPY --from=builder $OPT $OPT

## USER CONFIGURATION
RUN adduser --disabled-password --gecos '' ubuntu && chsh -s /bin/bash && mkdir -p /home/ubuntu

USER    ubuntu
WORKDIR /home/ubuntu

CMD ["/bin/bash"]