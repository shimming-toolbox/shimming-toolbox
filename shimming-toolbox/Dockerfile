FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git build-essential

# Install ST
RUN mkdir /ST_code && \
    git clone https://github.com/shimming-toolbox/shimming-toolbox.git /ST_code

# Install python dependencies
RUN pip3 install -r ST_code/shimming-toolbox/requirements_st.txt
