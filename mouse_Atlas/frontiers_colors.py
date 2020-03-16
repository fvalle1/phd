import matplotlib.pyplot as plt
import numpy as np

cmap1 = plt.get_cmap("tab20")
cmap2 = plt.get_cmap("tab20b")
cmap3 = plt.get_cmap("tab20c")
cmap4 = plt.get_cmap("Accent")
cmap5 = plt.get_cmap("Dark2")

colors = np.concatenate((cmap1(np.arange(20)), cmap2(np.arange(20)), cmap3(np.arange(20)),cmap4(np.arange(6)),cmap5(np.arange(4))))
colors = {tissue: c for tissue, c in zip (np.load("all_org.npy"), np.concatenate((cmap1(np.arange(20)), cmap2(np.arange(20)), cmap3(np.arange(20)),cmap4(np.arange(8)))))}
colors["Mesenchymal Stem Cell Cultured"]=colors["Mesenchymal St-Cell Cultured"]
colors["Fetal Stomache"] = colors["Fetal Stomach"]
colors["Adipose Tissue"] = colors["Adipose"]


def get_color(tissue):
    return colors[tissue.replace("-", " ").replace("c kit","c-kit").replace(".", " ").replace("_"," ").replace("MammaryGland", "Mammary Gland")]