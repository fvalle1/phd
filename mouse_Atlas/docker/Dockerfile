FROM jupyter/datascience-notebook

COPY Mouse_exploration.ipynb /home/jovyan
COPY tissue.ipynb /home/jovyan
COPY GTEx.ipynb /home/jovyan
COPY frontiers_analysis.py /home/jovyan
COPY frontiers_colors.py /home/jovyan
COPY tacos.jl /home/jovyan
COPY all_org.npy /home/jovyan
COPY requirements.jl /home/jovyan

RUN julia /home/jovyan/requirements.jl

WORKDIR /home/jovjan


ENTRYPOINT jupyter lab
