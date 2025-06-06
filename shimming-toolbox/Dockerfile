FROM ubuntu:24.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git make curl unzip vim xvfb bzip2 build-essential python3 python3-pip \
                    libglib2.0-0 libgl1 libxrender1 libxkbcommon-x11-0 libdbus-1-3 && \
    apt-get clean

# Install FSL
RUN mkdir code && \
    curl -Ls https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh | sh -s && \
    mv ~/fsl /code/

# Install SCT
RUN mkdir -p code/sct && \
    git clone --branch 7.0 --depth 1 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git /code/sct && \
    cd /code/sct && \
    ./install_sct -y

# Install ST
# TODO : Fix installation of ST (updated version dcm2niix craches the process)
RUN mkdir -p code/st && \
    git clone --branch v1.2 --depth 1 https://github.com/shimming-toolbox/shimming-toolbox.git /code/st && \
    cd /code/st && \
    make install

# Activate ST environment
SHELL ["/bin/bash", "-c"]
RUN echo 'source /code/st/python/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate /code/st/python' >> ~/.bashrc

# Clean up to reduce image size
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /code/sct/.git /code/st/.git && \
    /code/st/python/bin/conda clean -afy
