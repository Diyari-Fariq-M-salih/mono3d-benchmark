FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libeigen3-dev \
    libgl1-mesa-dev \
    libglew-dev \
    libgtk2.0-dev \
    libsuitesparse-dev \
    pkg-config \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upstream DSO source is expected to be mounted or cloned later.
