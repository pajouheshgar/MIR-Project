from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from Code.Phase3.vectorizers import TfIdf, word2vec
from Code.Utils.Config import Config
import pandas as pnd
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Clustering")

class K_Means:
    def __init__(self, vectorizer, n_clusters=5, max_iter=300):
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

    def report(self):
        logger.info("Saving clusters into a file")
        pnd.DataFrame(data=self.clusters, index=self.idx).to_csv(Config.DATA_DIR + '/Phase3/k_means_' +
                                                                 self.vectorizer_name + '.csv',
                                                                 header=False)

class Gaussian_Mixture:
    def __init__(self, vectorizer, n_components=5, max_iter=300, covariance_type='diag'):
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

class Hierarchical_Clustering:
    def __init__(self, vectorizer, n_clusters=5, linkage='ward'):
        self.vectorizer_name = vectorizer.name
        self.vectors = vectorizer.vectors
        self.idx = vectorizer.idx

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.hierarchical_clustering = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)

        logger.info("Inferring clusters using Hierarchical clustering over vectors of " + self.vectorizer_name)
        self.clusters = self.hierarchical_clustering.fit_predict(self.vectors)


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
    tfidf = TfIdf(all_text, indices, sparse=False)
    word2vec = word2vec(all_text, indices)

    K_Means(tfidf).report()
    K_Means(word2vec).report()

    Gaussian_Mixture(tfidf).report()
    Gaussian_Mixture(word2vec).report()

    Hierarchical_Clustering(tfidf).report()
    Hierarchical_Clustering(word2vec).report()