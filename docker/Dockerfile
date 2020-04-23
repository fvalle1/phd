FROM jupyter/datascience-notebook

COPY requirements.jl /home/jovyan
RUN julia /home/jovyan/requirements.jl

USER root
RUN apt-get update
RUN apt-get install sudo apt-utils --yes
RUN apt-get install libboost-all-dev python3 python3-scipy python-numpy \
              libcgal-dev libcairo2-dev python-cairo libsparsehash-dev   \
	      autoconf-archive pkg-config --yes

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime
RUN apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get install ipython libgtk-3-dev libjs-mathjax pandoc --yes
RUN pip3 install gobject matplotlib pandas jupyter cairocffi
RUN pip3 install scanpy pandarallel regex tensorflow watermark gseapy 
RUN pip3 install tensorflow_probability

RUN apt-get install gnupg --yes
RUN echo "deb http://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list
RUN echo "deb-src http://downloads.skewed.de/apt/ bionic main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
RUN apt-get update && apt-get install python3-graph-tool python3-cairo --yes

RUN wget https://gdc.cancer.gov/system/files/authenticated%20user/0/gdc-client_v1.5.0_Ubuntu_x64.zip
RUN unzip gdc-client_v1.5.0_Ubuntu_x64.zip
RUN mv gdc-client /usr/local/bin
RUN rm -f gdc-client_v1.5.0_Ubuntu_x64.zip

RUN useradd -m -g users filippo

ENV PYTHONIOENCODING=utf8

USER jovyan
WORKDIR /home/jovyan/work

ENTRYPOINT jupyter lab
CMD [""]
