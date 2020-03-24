FROM jupyter/datascience-notebook

COPY requirements.jl /home/jovyan
RUN julia /home/jovyan/requirements.jl


WORKDIR /home/jovjan/work


ENTRYPOINT jupyter lab
