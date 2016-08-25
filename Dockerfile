FROM ubuntu:16.04

RUN apt-get update && \
    apt-get -y install build-essential \
	gfortran \
	libblas-dev \
	liblapack-dev \
	libatlas-base-dev \
	python-dev \
	python-pip \
	libfreetype6-dev \
	libxft-dev \
	libpng12-dev  

ADD . /code
WORKDIR /code

RUN pip install --upgrade pip && \
	pip install virtualenv #&& \
#	virtualenv py && \
#	/code/py/bin/pip install -r requirements.txt

#ENTRYPOINT /bin/sh run.sh
#CMD /bin/bash
#CMD /code/py/bin/python main.py
