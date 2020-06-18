import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def plot_cox(fit_func):
	'''
	perform a fit using fit_func and create a plot if the p-values is low enough
	'''
	def wrapper(*args, **kwargs):
		summary, cph = fit_func(*args, **kwargs)
		if summary is None:
			return summary, cph, None
		name = summary.index[-1]
		p = float(summary.loc[name, "-log2(corrected_p)"])

		if p < 1:
			print(f"Too low -log2(p): {p}")
			return summary, cph, None
		ax = cph.plot_covariate_groups(name, [0,1], cmap='coolwarm', lw=10, figsize=(10,15))
		format_ax(ax, name, p)
		return summary, cph, ax

	return wrapper

@plot_cox
def fit_cox(subset, name, duration_col='days_survival', event_col='vital_status', *args, **kwargs):
	'''
	use lifelines to fit COXPHFitter model.
	return summary plus the corrected p-value

	subset: DataFrame
	name: name of the analysis
	duration_col: column of subset with number of days sample survived
	event_col: column of subset with 0/1 wheter the sample is alive or dead
	*args: to be passed to CoxPHFitter
	**kwargs: to be passed to CoxPHFitter
	'''
	from lifelines import CoxPHFitter
	from statsmodels.stats.multitest import multipletests
	cph = CoxPHFitter(*args, **kwargs)
	try:
		cph.fit(subset, duration_col=duration_col, event_col=event_col)
		summary = cph.summary
		p_vals = multipletests(cph.summary["p"], method="bonferroni")[1]
		summary["corrected_p"]=p_vals
		summary["-log2(corrected_p)"]=-np.log2(p_vals)
		summary.index.values[3]=name.replace(" ","_")
		return summary, cph
	except:
		print(*sys.exc_info())
		return None, None


def add_group_to_subset(topic: str, subset: pd.DataFrame, df_clusters: pd.DataFrame, quantile=0.75)->pd.DataFrame:
	'''
	add a column to subset with name topic
	topic: name
	subset: DataFrame
	df_clusters: DataFrame with p(sample|topic) on columns
	quantile: where to cut the binary annotation
	'''
	ret_subset = subset.copy()
	mask = df_clusters[topic]>df_clusters[topic].quantile(quantile)
	up_samples = df_clusters[mask].index
	ret_subset["group"] = np.zeros(ret_subset.shape[0])
	ret_subset.loc[ret_subset.index.isin(up_samples),["group"]]=1
	ret_subset["group"]=ret_subset["group"].astype(int)
	return ret_subset

def format_ax(ax, name = "", p = 1.) -> None:
    ax.set_title(f"Survival per {name}", fontsize=35)
    ax.set_xlabel("timeline (years from diagnosis)", fontsize=35)
    ax.set_ylabel("Survival", fontsize=35)

    lab = np.round(ax.get_xticks()/365).astype(int)
    ax.set_xticklabels(lab)
    ax.tick_params(labelsize=35)
    ax.set_title("-Log2(P_val): %.2f"%p, fontsize=35)

    for line in ax.get_lines():
        line.set_linewidth(10)
        label = line._label
        line.set_label(label.replace("=0", " down").replace("=1", " up"))

    ax.legend(fontsize=35)
    plt.tight_layout()

def save_plot(ax, dataset, topic) -> None:
	'''
	format and save survival_{dataset}_{topic}.pdf
	'''
	format_ax(ax)
	ax.get_figure().savefig(f"survival_{dataset}_{topic}.pdf")
