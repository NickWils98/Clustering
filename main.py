import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.cluster import BisectingKMeans, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from yellowbrick.cluster import SilhouetteVisualizer

NUMBER = 0


def find_optimal_eps(n_neighbors, data):
    """
    Find optimal epsilon distance by finding closest neighbour for each data point.
    Based on https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc

    Args:
        n_neighbors (int): number of neighbors to use
        data (pandas.core.frame.DataFrame): vectorized articles
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    plt.title("Find optimal epsilon")
    plt.xlabel("epsilon")
    plt.ylabel("number of articles")
    plt.plot(distances)
    plt.savefig('optimal_eps.png')
    plt.close()


def silhouette_analysis(K, vectorized_articles):
    """
    Silhouette analysis for a clustering.

    Args:
        K_list (range): number of clusters
        vectorized_articles (scipy.sparse.csr.csr_matrix): vectorized articles
    """
    cluster_algo = BisectingKMeans(n_clusters=K, random_state=NUMBER)
    #cluster_algo = KMeans(n_clusters=K, random_state=NUMBER)
    #cluster_algo = KMedoids(n_clusters=K, metric="cosine")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    sv = SilhouetteVisualizer(cluster_algo, colors="yellowbrick", ax=ax)
    ax.set_title(f"Silhouette analysis with {K} Cluster")
    sv.fit(vectorized_articles)
    print(f"cluster {K} with score {sv.silhouette_score_}")
    fig.savefig("silhouette_analysis.png")
    plt.close()


def elbow_test(K_list, vectorized_articles):
    """
    Elbowtest for a K cluster algorithm.

    Args:
        K_list (range): range between what K the elbowtest is done
        vectorized_articles (scipy.sparse.csr.csr_matrix): vectorized articles
    """
    inertia_list = []
    for K in K_list:
        cluster_algo = BisectingKMeans(n_clusters=K, random_state=NUMBER)
        #cluster_algo = KMeans(n_clusters=K, random_state=NUMBER)
        #cluster_algo = KMedoids(n_clusters=K, metric="cosine")
        cluster_algo.fit(vectorized_articles)
        inertia_list.append(cluster_algo.inertia_)

    # Plot the elbowtest
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    sns.lineplot(y=inertia_list, x=K_list, ax=ax).set(title="Elbowtest")
    ax.set_xlabel("Cluster count")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(K_list))
    fig.savefig("elbow_test.png")
    plt.close()


def filter_alpha_num(row):
    """
    Filter non-alphanumerical symbols and lower the text.

    Args:
        row (pandas.core.series.Series): row in matrix with columns [0 (titles),1 (articles)]

    Returns:
        string: preprocessed data
    """
    try:
        text = re.sub(r'[^ a-zA-Z0-9_ ]+', '', row[1])
        text = text.lower()
        return text
    except:
        print(row[1])


def print_top_words(feature_names, vectorized_articles, cluster_labels):
    """
    Print the top differentiating words and number of articles for each cluster.

    Args:
        feature_names (numpy.ndarray): array of the words
        vectorized_articles (scipy.sparse.csr.csr_matrix): vectorized articles
        cluster_labels (numpy.ndarray): array with a label for each article
    """
    df = pd.DataFrame(vectorized_articles.todense()
                      ).groupby(cluster_labels).mean()
    count = 0
    print(f"Cluster count = {df.shape[0]}\n")
    for i, r in df.iterrows():
        print(f"Cluster {count}")
        count += 1
        print(
            f"Number of articles in the cluster: {np.count_nonzero(cluster_labels == i)}")
        print(', '.join([feature_names[feature]
              for feature in np.argsort(r)[-5:]]))
        print("\n")


def make_dendagram(vectorized_articles):
    """
    Make a dendrogram from the articles.

    Args:
        vectorized_articles (scipy.sparse.csr.csr_matrix): vectorized articles
    """
    plt.title("Dendrogram of the articles")
    shc.dendrogram(shc.linkage(vectorized_articles.toarray(),
                   method='ward', metric="euclidean"))

    plt.savefig('dendrogram.png')
    plt.close()


if __name__ == "__main__":
    # Load articles
    filename = "articles.tsv"
    df = pd.read_csv(filename, sep='\t', header=None)

    # Preprocess the data
    df[2] = df.apply(filter_alpha_num, axis=1)
    articles = df[2].to_numpy()
    vectorizer = TfidfVectorizer(stop_words="english", min_df=10, max_df=0.90)
    vectorized_articles = vectorizer.fit_transform(articles)
    feature_names = vectorizer.get_feature_names_out()

    # Reduce dimensionality
    pca = PCA(n_components=2, whiten=False, random_state=NUMBER)
    pca_vectorized = pca.fit_transform(vectorized_articles.toarray())
    pca_vectorized = pd.DataFrame(data=pca_vectorized, columns=["x", "y"])

    # Get the optimal number of clusters
    make_dendagram(vectorized_articles)
    elbow_test(range(1, 40, 1), vectorized_articles)
    silhouette_analysis(14, vectorized_articles)
    find_optimal_eps(2, pca_vectorized)

    num_clusters = 14

    # Choose cluster algorithm and fit the data
    cluster_algo = BisectingKMeans(n_clusters=num_clusters, random_state=NUMBER)
    #cluster_algo = KMeans(n_clusters=num_clusters, random_state=NUMBER)
    #cluster_algo = KMedoids(n_clusters=num_clusters, metric="cosine")
    #cluster_algo = DBSCAN(eps=0.01, metric="l2")

    cluster_algo.fit(vectorized_articles)

    # AgglomerativeClustering needs the data to be an array
    #cluster_algo = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')
    #cluster_algo.fit(vectorized_articles.toarray())

    # Labels of the clusters
    cluster_labels = cluster_algo.labels_

    # add titles and labels to the df
    pca_vectorized["cluster_labels"] = cluster_labels
    pca_vectorized["Titles"] = df[0]

    # Plot the clusters
    plt.title("Clustering of the articles")
    plt.xlabel("y")
    plt.ylabel("x")
    plot = sns.scatterplot(data=pca_vectorized, x="x",
                           y="y", hue="cluster_labels")
    fig = plot.get_figure()
    fig.savefig("clustering.png")
    plt.show()
    plt.close()

    # Print the top 5 differentiating words
    print_top_words(feature_names, vectorized_articles, cluster_labels)
    # Write to clusters.tsv
    #pca_vectorized[['Titles', 'cluster_labels']].to_csv('clusters.tsv', sep="\t", header=False, index=False)
