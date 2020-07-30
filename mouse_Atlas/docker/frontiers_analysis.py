import os, gc
import numpy as np
import pandas as pd
from scipy import stats

def get_files():
    # ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE108nnn/GSE108097/matrix/
    os.system("wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE108nnn/GSE108097/matrix/GSE108097-GPL17021_series_matrix.txt.gz")
    os.system("gunzip GSE108097-GPL17021_series_matrix.txt.gz")
    os.system("mv GSE108097-GPL17021_series_matrix.txt mca/.")
    os.system("mkdir -p mca/data")
    with open("mca/GSE108097-GPL17021_series_matrix.txt", "r") as file:
        os.chdir("mca")
        for row in file.readlines():
            if "Sample_supplementary_file_1" in row:
                for sample in row.split("\t")[1:-1]:
                    print(sample)
                    os.system(f"wget {sample}")
    os.chdir("../")
    cleanup()

def cleanup():
    os.chdir("mca")
    for file in os.listdir():
        print(file)
        if (".ipynb_checkpoints" in file) or ("DS_Store" in file) or ("data" in file):
            continue
        if ".gz" in file:
            os.system(f"gunzip {file}")
            unpacked_file = file[:-3]
        else:
            unpacked_file = file
        try:
            with open(unpacked_file, "r") as f:
                data = f.read().replace("\"","")
            with open(unpacked_file, "w") as f:
                f.write(data)
        except:
            print("Error with %s"%unpacked_file)
    os.chdir("../")

def load_pickle(filename):
    import pickle
    with open(filename,"rb") as f:
        data = pickle.load(f)
    return data


def heaps(M, diffWords, tissue,  fit_bins = lambda x, a, b: a * np.power(x,b)):
    from scipy.optimize import curve_fit
    if len(M) < 2:
        return
    fig = plt.figure(figsize=(10,6))
    plt.title(tissue)
    plt.scatter(M, diffWords, label='samples')
    #plt.scatter(M_null, diffWords_null, label='null_model')
    plt.xlabel("Transcriptome size", fontsize=24)
    plt.ylabel("Number of\n expressed genes", fontsize=24)
    n_bins=35
    bin_means, bin_edges, binnumber = stats.binned_statistic(M, diffWords,statistic='mean', bins=np.linspace(M.min(),max(M), n_bins))
    bin_counts, _, _ = stats.binned_statistic(M, diffWords,statistic='count', bins=np.linspace(M.min(),max(M), n_bins))


    skip_bins=(bin_counts<10).astype(int).sum()
    x_bins = ((bin_edges[:-1]+bin_edges[1:])/2)[:-skip_bins]

    if len(bin_means) - skip_bins < 2:
        return

    plt.hlines(bin_means[:-skip_bins], bin_edges[:-1][:-skip_bins], bin_edges[1:][:-skip_bins], colors='r', lw=5, label='binned average')
    bin_stds, _, _ = stats.binned_statistic(M, diffWords,statistic='std',  bins=np.linspace(M.min(),max(M), n_bins))
    plt.errorbar(x_bins,bin_means[:-skip_bins], bin_stds[:-skip_bins], fmt='none', ecolor='orange', elinewidth=3)

    popt, pcov = curve_fit(fit_bins, x_bins, bin_means[:-skip_bins])

    plt.plot(np.linspace(500, 1e4), fit_bins(np.linspace(500, 1e4), *popt), lw=4, c='cyan')

    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.0E'))

    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(0,M.max()+500)
    plt.ylim(0,diffWords.max()+500)
    plt.legend(fontsize=20)
    plt.show()
    fig.savefig(f"heaps_{tissue}.png")


def save_model(df, name, tissue="global", n_bins=35, fit_bins = lambda x, a, b: a * np.power(x,b), **kwargs):
    from scipy.optimize import curve_fit
    import pickle

    print("saving")
    from scipy.integrate import quad
    A = df.mean(axis=1)
    A = A[A>0]
    df = df.reindex(index=A.index)
    f = A/A.sum()
    O = df.apply(lambda x: len(x[x>0])/float(len(x)), 1)
    M = df.apply(np.sum, 0)
    diffWords = df.apply(lambda x: len(x[x>0]), 0)
    means = df.apply(np.mean, 1)
    var = df.apply(np.var, 1)
    cv2= var/means/means

    try:
        bin_means, bin_edges, binnumber = stats.binned_statistic(M, diffWords,statistic='mean', bins=np.linspace(M.min(),max(M), n_bins))
        bin_counts, _, _ = stats.binned_statistic(M, diffWords,statistic='count', bins=np.linspace(M.min(),max(M), n_bins))

        skip_bins=(bin_counts<10).astype(int).sum()

        if len(bin_means) - skip_bins < 2:
            x_bins = ((bin_edges[:-1]+bin_edges[1:])/2)[:-skip_bins]
            popt, pcov = curve_fit(fit_bins, x_bins, bin_means[:-skip_bins])
            integral = quad(fit_bins, 500, 1e4, args=(popt[0], popt[1]))
        else:
            popt = []
            integral = -1
    except:
        popt = []
        integral = -1

    data = {
    'means': means,
    'means_nonzero': df.apply(lambda x: x[x>0].mean(), 1),
    'var': var,
    'freq': f,
    'O': O,
    'M': M,
    'cv2': cv2,
    'diffWords': diffWords,
    'heaps_integral': integral,
    'heaps_fit': popt
    }

    with open(f"data_{tissue}_{name}.pkl","wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_tissue(tissue, name="data", data_source="mca"):
    import pickle
    with open(f"{data_source}/data_{tissue}_{name}.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def mazzolini(M, f, tissue, ensambles = 5, **kwargs):
    print("mazzolini")
    print(f"sampling {len(M)} cells")
    global rs
    rs = np.random.RandomState(seed=42)
    import pickle
    data = {
    'means': [],
    'means_nonzero':[],
    'var': [],
    'freq': [],
    'O': [],
    'M': [],
    'cv2': [],
    'diffWords': [],
    'heaps_integral': [],
    'heaps_fit': []
    }

    for ensamble in range(ensambles):
        df_null = pd.DataFrame(index=f.index)
        for isample,sample in enumerate(M.index):
            if (isample % (len(M)/1000)) == 0:
                print(f"{isample} of {len(M)}")
            if sample in df_null.columns:
                continue
            df_null.insert(0,sample,rs.multinomial(M[sample], f.astype(float).values/f.sum()))
            gc.collect()

        A = df_null.mean(axis=1)
        A = A[A>0]
        df = df_null.reindex(index=A.index)
        data["freq"].append(A/A.sum())
        data["O"].append(df_null.apply(lambda x: len(x[x>0])/float(len(x)), 1))
        data["M"].append(df_null.apply(np.sum, 0))
        data["diffWords"].append(df_null.apply(lambda x: len(x[x>0]), 0))
        var = np.array(df_null.apply(np.var, 1))
        means = np.array(df_null.apply(np.mean, 1))
        data["means"].append(means)
        data["means_nonzero"].append(df_null.apply(lambda x: x[x>0].mean(), 1))
        data["var"].append(var)
        data["cv2"].append(var/means/means)
        del df_null
        gc.collect()

    for key, value in data.items():
        data[key]=np.average(value, axis=0)


    #df_null=df_null.astype(int)
    gc.collect()
    with open(f"data_{tissue}_mazzolini.pkl","wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    gc.collect()

def null_model(M, f, tissue, **kwargs)->None:
    import pickle
    print("null_1")
    global rs

    data = {
    'means': [],
    'means_nonzero':[],
    'var': [],
    'freq': [],
    'O': [],
    'M': [],
    'cv2': [],
    'diffWords': [],
    'heaps_integral': [],
    'heaps_fit': []
    }

    rs = np.random.RandomState(seed=42)

    for ensamble in range(3):
        df_null = pd.DataFrame(index=f.index)
        for isample,sample in enumerate(M.index):
            if (isample % (len(M)/25)) == 0:
                print(f"{isample} of {len(M)}")
            if sample in df_null.columns:
                continue
            df_null.insert(0,sample,rs.multinomial(M[sample], f.astype(float).values/f.sum()))
            gc.collect()

        A = df_null.mean(axis=1)
        A = A[A>0]
        df = df_null.reindex(index=A.index)
        data["freq"].append(A/A.sum())
        data["O"].append(df_null.apply(lambda x: len(x[x>0])/float(len(x)), 1))
        data["M"].append(df_null.apply(np.sum, 0))
        data["diffWords"].append(df_null.apply(lambda x: len(x[x>0]), 0))
        var = np.array(df_null.apply(np.var, 1))
        means = np.array(df_null.apply(np.mean, 1))
        data["means"].append(means)
        data["means_nonzero"].append(df_null.apply(lambda x: x[x>0].mean(), 1))
        data["var"].append(var)
        data["cv2"].append(var/means/means)
        del df_null
        gc.collect()

    for key, value in data.items():
        data[key]=np.average(value, axis=0)


    #df_null=df_null.astype(int)
    gc.collect()
    with open(f"data_{tissue}_null.pkl","wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    gc.collect()

def load_all_data(data_source="mca"):
	import pickle
	with open(f"{data_source}/data_all.pkl","rb") as file:
		data = pickle.load(file)

	means = data['means']
	var = data['var']
	f = data['freq']
	O = data['O']
	M = data["M"]
	cv2 = data['cv2']
	diffWords = data['diffWords']
	means_nozero = data['means_nonzero']
	if 'n_expressed' in data.keys():
		n_expressed_genes = data['n_expressed']
	if 'n_genes' in data.keys():
		n_genes = data['n_genes']
	if 'frac_of' in data.keys():
		frac_of = data['frac_of']
	if 'cell_zeros' in data.keys():
		cell_zeros = data['cell_zeros']
	return data

# Tabula muris only!
def clean_df(df):
    dfann = pd.read_csv("annotations_facs.csv")
    bad_cells = [c for c in df.columns if c not in dfann["cell"].values]
    df = df.drop(columns=bad_cells)
    genes_not_si=filter(lambda g: not 'ERCC' in g, df.index)
    df = df.reindex(index=list(genes_not_si))
    return df
