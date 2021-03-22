FROM jupyter/datascience-notebook:latest

LABEL maintainer="filippo.valle@unito.it"

USER root
RUN apt-get update
RUN apt-get install sudo apt-utils --yes

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime
RUN apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

USER root

USER jovyan
RUN python3 -m pip install --no-cache-dir jupyterlab==2.2.9
RUN jupyter labextension install @jupyterlab/toc --no-build
RUN jupyter labextension install jupyterlab-plotly --no-build
RUN jupyter labextension install plotlywidget --no-build
RUN jupyter lab build

USER root
RUN apt-get install xvfb --yes
RUN apt-get autoremove --yes

ENV PYTHONIOENCODING=utf8

USER jovyan
RUN conda config --set channel_priority strict

RUN conda install -c conda-forge mamba -y
RUN mamba install -c plotly plotly-orca -y
RUN mamba install -c conda-forge graph-tool -y

COPY requirements.txt /home/jovyan
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /home/jovyan/work

ENTRYPOINT jupyter lab
CMD [""]
