FROM jupyter/datascience-notebook

COPY requirements.jl /home/jovyan
RUN julia /home/jovyan/requirements.jl

COPY requirements.txt /home/jovyan
RUN python3 -m pip install -U -r /home/jovyan/requirements.txt


RUN mkdir -p /home/jovyan/work
WORKDIR /home/jovyan/work
COPY frontiers_analysis.py /home/jovyan/work
COPY run_mazzolini.py /home/jovyan/work

ENTRYPOINT python3 run_mazzolini.py
#ENTRYPOINT jupyter lab
