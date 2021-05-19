FROM ubuntu:16.04

ENV DISPLAY="$DISPLAY"
ENV HOME="/root"
ENV EVALUATION_DIR="${HOME}/local-features-evaluation"
ENV DEPENDENCIES_DIR="${HOME}/dependencies"
ENV OPENCV_DIR="${DEPENDENCIES_DIR}/opencv"
ENV OPENCV_CONTRIB_DIR="${DEPENDENCIES_DIR}/opencv_contrib"

# Install dependencies

# Update package list and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    checkinstall \
    x11-apps && \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# clone opencv 3.4.1
WORKDIR ${DEPENDENCIES_DIR}
RUN git clone https://github.com/opencv/opencv.git opencv && \
    cd opencv && \
    git checkout 3.4.1

# clone opencv_contrib 3.4.1
WORKDIR ${DEPENDENCIES_DIR}
RUN git clone https://github.com/opencv/opencv_contrib.git opencv_contrib && \
    cd opencv_contrib && \
    git checkout 3.4.1

# compile opencv with opencv_contrib module
WORKDIR ${OPENCV_DIR}
RUN mkdir build && \
    cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_DIR}/modules .. && \
    make -j && \
    make install

# Add Evaluation directory to docker
COPY ./ ${EVALUATION_DIR}
WORKDIR ${EVALUATION_DIR}
RUN mkdir build && \
    cd build && \
    cmake .. && \
    make -j

# Go to the executable directory
WORKDIR ${EVALUATION_DIR}/build/mainproject

# in order to leave running the terminal we should leave running bash
RUN ["/bin/bash"]
