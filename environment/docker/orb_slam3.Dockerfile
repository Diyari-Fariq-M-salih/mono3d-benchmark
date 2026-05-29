FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libgl1-mesa-dev \
    libglew-dev \
    libgtk2.0-dev \
    libopencv-dev \
    libpython3-dev \
    pkg-config \
    python3 \
    python3-pip \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upstream Pangolin and ORB-SLAM3 source are expected to be mounted or cloned later.
