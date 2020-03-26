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
        
        
def mazzolini(M, f, tissue, **kwargs):
    print("mazzolini")
    global rs 
    rs = np.random.RandomState(seed=42)
    df_null = pd.DataFrame(index=f.index)
    for sample in M.index:
        if sample in df_null.columns:
            continue
        df_null.insert(0,sample,np.average(np.array([rs.multinomial(M[sample], f.astype(float).values/f.sum()) for _ in range(5)]), axis=0))
        gc.collect()
    #df_null=df_null.astype(int)
    gc.collect()
    save_model(df_null, "mazzolini", tissue, **kwargs)
    del df_null
    gc.collect()
        
def null_model(df, M, means_nozero):
    print("null_model: start")
    global rs
    rs = np.random.RandomState(seed=42)
    f_null_1 = (means_nozero / means_nozero.sum()).dropna()
    df_null_1 = pd.DataFrame(index=f_null_1.index)
    number_of_zeros = df.apply(lambda x: len(x[x==0]), 1)
    for sample in M.index:
        if sample in df_null_1.columns:
            continue
        df_null_1.insert(0,sample,np.average(np.array([rs.multinomial(M[sample], f_null_1.astype(float).values) for _ in range(2)]), axis=0))
        gc.collect()
    df_null_1=df_null_1.round()
    #df_null_1 = df_null_1.divide(df_null_1.sum(0),1).multiply(M[df_null_1.columns])
    gc.collect()
    
    print("null model: flipping 0s")
    
    number_of_sampled_zeros = df_null_1.apply(lambda x: len(x[x==0]), 1)
    df_null_1 = df_null_1.transpose()

    quantiles = np.quantile(M.sort_values(), q=np.linspace(0,1,15)[:-1])
    classes = np.digitize(M.sort_values(),quantiles)

    genes_with_many_0 = []
    for g in df_null_1.columns:
        gexpr = df.loc[g,:]
        if number_of_zeros[g] > number_of_sampled_zeros[g]:
            df_null_1[g][rs.choice(df_null_1[g][df_null_1[g]>0].index, size=number_of_zeros[g]-number_of_sampled_zeros[g], replace=False)]=0
        else:
            genes_with_many_0.append(g)
        #df_null_1[g][rs.choice(df_null_1[g].index, size=number_of_zeros[g], replace=False)]=0
        #prob_classes = [(gexpr[M.index[classes==c]]==0).astype(int).sum()/float(len(M.index[classes==c])) for c in np.arange(max(classes))+1]
        #probs = pd.Series(index=M.index, data=np.zeros_like(M))
        #for c, prob_class in zip(classes, prob_classes):
        #    probs[M.index[classes==c]] = prob_class
        #df_null_1[g][rs.binomial(1,p=probs, size=len(df_null_1[g]))==1] = 0
        #del probs
        del gexpr
        gc.collect()
        
    df_null_1 = df_null_1.transpose()
    
    #df_null_1 = df_null_1.applymap(lambda x: 0 if rs.random()< np.exp(-x) else x)
    df_null_1 = df_null_1.divide(df_null_1.sum(0),1).multiply(M[~M.duplicated()][df_null_1.columns])
    gc.collect()
    save_model(df_null_1, "null_1", tissue)
    with open(f"many0genes_{tissue}.txt","w") as file:
        for g in genes_with_many_0:
            file.write(g+"\n")
    del df_null_1
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