FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libblas-dev \
    libeigen3-dev \
    libglew-dev \
    liblapack-dev \
    libopencv-dev \
    libsuitesparse-dev \
    libtool \
    libyaml-cpp-dev \
    lsb-release \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1.list' \
    && wget -qO - https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update \
    && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    && pip3 install --no-cache-dir catkin-tools vcstool osrf-pycommon \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-lc"]

# Usage:
#   docker build -f environment/docker/svo.Dockerfile -t mono3d-benchmark/svo:noetic .
#   docker run --rm -it -v "$PWD":/workspace mono3d-benchmark/svo:noetic bash
