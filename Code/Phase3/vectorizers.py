import logging

import pandas as pnd
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.sklearn_api
from Code.Utils.Config import Config

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Vectorizer")


class TfIdf:
    def __init__(self, corpus, idx, sparse=True):
        self.name = 'tfidf'
        self.corpus = corpus
        self.idx = idx

        logger.info("Inferring tf-idf from data")
        self.vectorizer = TfidfVectorizer(max_df=Config.MAX_DF, max_features=Config.MAX_TF_IDF_FEATURES)
        self.vectors = self.vectorizer.fit_transform(self.corpus)

        if not sparse:
            self.vectors = self.vectors.toarray()


    def tfidf_tranform(self, doc, sparse):
        transforemd = self.vectorizer.transform(doc)
        if sparse:
            return transforemd
        else:
            return transforemd.toarray()[0, :]


class Word2Vec:
    def __init__(self, corpus, idx):
        self.corpus = corpus
        self.idx = idx
        logger.info('Inferring word2vec from data')

        self.vectorizer = TfidfVectorizer(max_df=Config.MAX_DF, max_features=Config.MAX_TF_IDF_FEATURES)
        self.vectors = self.vectorizer.fit_transform(self.corpus)

        if not sparse:
            self.vectors = self.vectors.toarray()


if __name__ == '__main__':
    data = pnd.read_csv(Config.CLUSTERING_DATA_DIR, encoding='latin1', index_col=0)
    all_text = data.values
    all_text = [text for sublist in all_text for text in sublist]
    indices = data.index.values
    tfidf = TfIdf(all_text, indices)
