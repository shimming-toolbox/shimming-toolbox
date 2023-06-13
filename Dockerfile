FROM ubuntu:latest
ADD . /src/fsleyes-plugin-shimming-toolbox/

RUN  apt-get update \
  && apt-get install -y make vim freeglut3 gcc curl unzip xvfb

RUN cd /src/fsleyes-plugin-shimming-toolbox && make install

RUN ~/shimming-toolbox/python/bin/conda init bash
