FROM jupyter/datascience-notebook
ADD install_pkgs.R /home/jovyan/work/

USER root
RUN apt-get update && apt-get upgrade -y
RUN apt-get install libicu-dev apt-utils -y

USER jovyan 

WORKDIR /home/jovyan/work/
RUN Rscript install_pkgs.R


