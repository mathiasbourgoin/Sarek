FROM intel/oneapi-runtime:2024.2.1-0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG CACHEBUST
RUN echo "cache bust: ${CACHEBUST}" > /tmp/cachebust

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    clang \
    curl \
    git \
    libffi-dev \
    libx11-dev \
    m4 \
    opam \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

# Intel oneAPI already provides OpenCL runtime
ENV OPAMYES=1
