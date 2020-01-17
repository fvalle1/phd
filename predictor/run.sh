#!/bin/bash
source /opt/intel/openvino/bin/setupvars.sh
#cd predictor && python3.7 run.py
jupyter-notebook --ip 0.0.0.0 --allow-root --port 8899
