FROM python:3.8

ARG dev_user=natadev
ARG home_dir=/home/${dev_user}
ARG work_dir=${home_dir}/nata

# SZIP
ARG SZIP_VERSION=2.1.1
ARG SZIP_SHA256=21ee958b4f2d4be2c9cabfa5e1a94877043609ce86fde5f286f105f7ff84d412
ENV SZIP_DIR=/usr/local/szip

# HDF5
ARG HDF5_SHORT_VER=1.12
ARG HDF5_VERSION=1.12.0
ARG HDF5_SHA256=97906268640a6e9ce0cde703d5a71c9ac3092eded729591279bf2e3ca9765f61
ENV HDF5_DIR=/usr/local/hdf5

# compile and install HDF5 tooling
RUN \
  wget https://support.hdfgroup.org/ftp/lib-external/szip/${SZIP_VERSION}/src/szip-${SZIP_VERSION}.tar.gz \
  && echo "${SZIP_SHA256} szip-${SZIP_VERSION}.tar.gz" | sha256sum -c \
  && tar -xzf szip-${SZIP_VERSION}.tar.gz \
  && cd szip-${SZIP_VERSION} \
  && ./configure --prefix=${SZIP_DIR} \
  && make \
  && make install \
  && cd .. \
  && rm -rf szip-${SZIP_VERSION} szip-${SZIP_VERSION}.tar.gz \
  && wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_SHORT_VER}/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.bz2 \
  && echo "${HDF5_SHA256} hdf5-${HDF5_VERSION}.tar.bz2" | sha256sum -c \
  && tar -xf hdf5-${HDF5_VERSION}.tar.bz2 \
  && cd hdf5-${HDF5_VERSION} \
  && ./configure --prefix=${HDF5_DIR} \
  && make \
  && make install \
  && cd .. \
  && rm -rf hdf5-${HDF5_VERSION} hdf5-${HDF5_VERSION}.tar.bz2

# developer tooling and runtime package
RUN \
  apt-get update \
  && apt-get install -y texlive-base graphviz less \
  && pip install poetry black pre-commit

# Setup project directory & poetry
RUN \
  adduser \
  --disabled-password \
  --home ${home_dir} \
  "${dev_user}"

USER ${dev_user}
WORKDIR ${work_dir}
