FROM ubuntu:latest

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git make build-essential curl unzip vim xvfb python3 python3-pip

# Install ST
RUN mkdir -p /code/ST && \
    git clone https://github.com/shimming-toolbox/shimming-toolbox.git /code/ST  && \
    cd /code/ST && \
    make install
