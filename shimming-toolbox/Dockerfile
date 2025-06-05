FROM ubuntu:latest

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git make curl unzip vim xvfb bzip2 build-essential \
                    python3 python3-pip libglib2.0-0 libgl1 libxrender1 libxkbcommon-x11-0 libdbus-1-3 && \
    apt-get clean

# Install FSL
RUN curl -Ls https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh | sh -s

# Install SCT
RUN mkdir -p code/SCT && \
    git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git /code/SCT && \
    cd /code/SCT && \
    ./install_sct -y

# Install ST
RUN cd .. && \
    git clone https://github.com/shimming-toolbox/shimming-toolbox.git /code/ST  && \
    cd /code/ST && \
    make install

# Acivate ST environment
SHELL ["/bin/bash", "-c"]
RUN echo 'source $HOME/shimming-toolbox/python/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate ~/shimming-toolbox/python' >> ~/.bashrc

# Clean up to reduce image size
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /code/SCT/.git /code/ST/.git && \
    ~/shimming-toolbox/python/bin/conda clean -afy
