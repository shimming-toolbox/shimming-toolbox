FROM ubuntu:latest
ADD . /src/fsleyes-plugin-shimming-toolbox/


RUN  apt-get update \
  && apt-get install -y make vim freeglut3 gcc

RUN cd /src/fsleyes-plugin-shimming-toolbox
