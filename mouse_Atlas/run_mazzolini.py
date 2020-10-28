from frontiers_analysis import mazzolini
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os, sys, gc
import pickle

import itertools
import seaborn as sns

from frontiers_analysis import heaps, load_tissue, save_model, mazzolini

print("loaded libraries")

tissue = "Bone-Marrow_c-Kit"

import os
print(os.listdir("mca"))
with open(f"{data_source}/data_{tissue}_{name}.pkl", "rb") as f:
    data = pickle.load(f)

means = data['means']
var = data['var']
f = data['freq']
O = data['O']
M = data["M"]
cv2 = data['cv2']
diffWords = data['diffWords']
means_nozero = data['means_nonzero']

print("loaded data")

mazzolini(M, f, tissue)

for method in ["null", "nullteo"]:
    try:
        data[method] = load_tissue(tissue, method)

        means_null = data[method]['means']
        var_null = data[method]['var']
        f_null = data[method]['freq']
        O_null = data[method]['O']
        M_null = data[method]["M"]
        cv2_null = data[method]['cv2']
        diffWords_null = data[method]['diffWords']
        means_nozero_null = data[method]['means_nonzero']
    except:
        print(*sys.exc_info())
