from sklearn.metrics.cluster import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
from topicpy.hsbmpy import color_iterator
from matplotlib import cm
import logging
log = logging.getLogger("silhouette")
hdl = logging.StreamHandler()
hdl.setLevel(logging.INFO)
log.addHandler(hdl)

def make_silhouette(data, label, classes, cluster_labels, n_clusters, metrics = ["euclidean", "cosine"]):
    for k in metrics:
        log.info("metric "+k)
        y_lower = 10
        sample_silhouette_values = silhouette_samples(data, cluster_labels, metric=k)
        silhouette_avg=silhouette_score(data, cluster_labels, metric=k)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 35)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, data.shape[0] + (n_clusters + 1) * 10])

        for i in range(n_clusters):
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            #color = cm.gnuplot(float(i) / n_clusters)
            color = next(color_iterator)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.8, (y_lower+y_upper)/2, "%s"%classes[i], fontsize=18)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples


        ax.set_title("gtex %s\n%s metric\n%d clusters\n%s space"%(label,k,n_clusters, label), fontsize=20)
        ax.set_xlabel("score", fontsize=20)
        # Label the silhouette plots with their cluster numbers at the middle
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", lw=2)
        plt.tick_params(labelsize=20)
        plt.show()
        fig.savefig("silhouette_%s_%s.pdf"%(label,k))
        del sample_silhouette_values
        del silhouette_avg