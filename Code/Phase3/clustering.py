from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from Code.Phase3.vectorizers import TfIdf, Word2Vec
from Code.Utils.Config import Config

import pandas as pnd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Clustering")


class K_Means:
    def __init__(self, vectorizer, n_clusters=3, max_iter=300):
        self.vectorizer_name = vectorizer.name
        self.vectors = vectorizer.vectors
        self.idx = vectorizer.idx

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.k_means = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)

        logger.info("Inferring clusters using k-means over vectors of " + self.vectorizer_name)
        self.clusters = self.k_means.fit_predict(self.vectors)

    def transform(self, vector):
        return self.k_means.transform(vector)

    def visualize(self, method='pca', n_iter=300):
        logger.info("Visualizing clusters")
        plt.figure(figsize=(6, 6))
        plt.title(
            method.upper() + ' with K-Means clusters on ' + self.vectorizer_name + ' (K = {})'.format(self.n_clusters))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        cluster_colors = [colors[cluster_num] for cluster_num in self.clusters]

        cluster_centers = self.k_means.cluster_centers_
        input_features = np.concatenate((self.vectors, cluster_centers), axis=0)
        if method == 'tsne':
            dim_reduction = TSNE(n_components=2, n_iter=n_iter)
        elif method == 'pca':
            dim_reduction = PCA(n_components=2)
        else:
            raise Exception

        transformed_features = dim_reduction.fit_transform(input_features)

        plt.scatter(x=transformed_features[:-self.n_clusters, 0],
                    y=transformed_features[:-self.n_clusters, 1],
                    c=cluster_colors,
                    s=4,
                    marker='o')

        plt.scatter(x=transformed_features[-self.n_clusters:, 0],
                    y=transformed_features[-self.n_clusters:, 1],
                    c='black',
                    s=60,
                    marker='x')

        plt.show()

    def report(self):
        logger.info("Saving clusters into a file")
        pnd.DataFrame(data=self.clusters, index=self.idx).to_csv(Config.DATA_DIR + '/Phase3/k_means_' +
                                                                 self.vectorizer_name + '.csv',
                                                                 header=False)


class Gaussian_Mixture:
    def __init__(self, vectorizer, n_components=3, max_iter=300, covariance_type='full'):
        self.vectorizer_name = vectorizer.name
        self.vectors = vectorizer.vectors
        self.idx = vectorizer.idx

        self.n_components = n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.gaussian_mixture = GaussianMixture(n_components=self.n_components,
                                                max_iter=self.max_iter,
                                                covariance_type=covariance_type)

        logger.info("Inferring clusters using Gaussian Mixture model over vectors of " + self.vectorizer_name)
        self.gaussian_mixture.fit(self.vectors)
        self.clusters = self.gaussian_mixture.predict(self.vectors)

    def transform(self, vector):
        return self.gaussian_mixture.predict(vector)

    def report(self):
        logger.info("Saving clusters into a file")
        pnd.DataFrame(data=self.clusters, index=self.idx).to_csv(Config.DATA_DIR + '/Phase3/Gaussian Mixture_' +
                                                                 self.vectorizer_name + '.csv',
                                                                 header=False)

    def visualize(self, method='pca', n_iter=300):
        logger.info("Visualizing clusters")
        plt.figure(figsize=(6, 6))
        plt.title(
            method.upper() + ' with Gaussian Mixture clusters on ' + self.vectorizer_name +
            ' (# of Components = {})'.format(self.n_components))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        cluster_colors = [colors[cluster_num] for cluster_num in self.clusters]

        cluster_centers = self.gaussian_mixture.means_
        input_features = np.concatenate((self.vectors, cluster_centers), axis=0)
        if method == 'tsne':
            dim_reduction = TSNE(n_components=2, n_iter=n_iter)
        elif method == 'pca':
            dim_reduction = PCA(n_components=2)
        else:
            raise Exception

        transformed_features = dim_reduction.fit_transform(input_features)

        plt.scatter(x=transformed_features[:-self.n_components, 0],
                    y=transformed_features[:-self.n_components, 1],
                    c=cluster_colors,
                    s=4,
                    marker='o')

        plt.scatter(x=transformed_features[-self.n_components:, 0],
                    y=transformed_features[-self.n_components:, 1],
                    c='black',
                    s=60,
                    marker='x')

        plt.show()


class Hierarchical_Clustering:
    def __init__(self, vectorizer, max_d, n_clusters=3, linkage='ward'):
        self.vectorizer_name = vectorizer.name
        self.vectors = vectorizer.vectors
        self.idx = vectorizer.idx

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.max_d = max_d

        logger.info("Inferring clusters using Hierarchical clustering over vectors of " + self.vectorizer_name)
        self.hierarchical_clustering = AgglomerativeClustering(n_clusters=3,
                                                               linkage=self.linkage)
        self.clusters = self.hierarchical_clustering.fit_predict(self.vectors)

    def plot_dendrogram(self):
        m = sch.linkage(self.vectors, method=self.linkage)
        sch.dendrogram(m, truncate_mode='lastp', p=20, leaf_rotation=90.)
        plt.axhline(y=self.max_d, c='k')
        plt.title('Truncated Dendrogram')
        plt.show()

    def visualize(self, method='pca', n_iter=300):
        logger.info("Visualizing clusters")
        plt.figure(figsize=(6, 6))
        plt.title(
            method.upper() + ' with Hierarchical Clustering clusters on ' + self.vectorizer_name +
            ' (# of Components = {})'.format(self.n_clusters))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        cluster_colors = [colors[cluster_num] for cluster_num in self.clusters]

        if method == 'tsne':
            dim_reduction = TSNE(n_components=2, n_iter=n_iter)
        elif method == 'pca':
            dim_reduction = PCA(n_components=2)
        else:
            raise Exception

        transformed_features = dim_reduction.fit_transform(self.vectors)

        plt.scatter(x=transformed_features[:, 0],
                    y=transformed_features[:, 1],
                    c=cluster_colors,
                    s=4,
                    marker='o')
        plt.show()

    def report(self):
        logger.info("Saving clusters into a file")
        pnd.DataFrame(data=self.clusters, index=self.idx).to_csv(Config.DATA_DIR + '/Phase3/Hierarchical_Clustering_' +
                                                                 self.vectorizer_name + '.csv',
                                                                 header=False)


if __name__ == '__main__':
    data = pnd.read_csv(Config.CLUSTERING_DATA_DIR, encoding='latin1', index_col=0)
    all_text = data.values
    all_text = [text for sublist in all_text for text in sublist]
    indices = data.index.values
    vectorizers = [TfIdf(all_text, indices, sparse=False),
                   Word2Vec(all_text, indices, n_epochs=50, dim=300)]

    for vectorizer in vectorizers:
        kmeans_tfidf = K_Means(vectorizer=vectorizer,
                               n_clusters=3,
                               max_iter=300)
        kmeans_tfidf.visualize(method='pca')
        kmeans_tfidf.report()

        gmm_tfidf = Gaussian_Mixture(vectorizer=vectorizer,
                                     n_components=3,
                                     max_iter=300,
                                     covariance_type='full')
        gmm_tfidf.visualize(method='pca')
        gmm_tfidf.report()

    vectorizers = [TfIdf(all_text, indices, sparse=False, max_features=5000),
                   Word2Vec(all_text, indices, n_epochs=50, dim=100)]
    for vectorizer in vectorizers:
        hierarchical = Hierarchical_Clustering(vectorizer=vectorizer,
                                               max_d=4 if vectorizer.name=='tfidf' else 40,
                                               n_clusters=3,
                                               linkage='ward')
        hierarchical.plot_dendrogram()
        hierarchical.visualize(method='pca')
        hierarchical.report()
