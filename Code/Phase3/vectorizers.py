import logging
import pandas as pnd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.sklearn_api import W2VTransformer
from gensim.utils import simple_preprocess
from Code.Utils.Config import Config

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Vectorizer")


class TfIdf:
    def __init__(self, corpus, idx, sparse=True):
        self.name = 'tfidf'
        self.corpus = corpus
        self.idx = idx

        logger.info("Inferring tf-idf from data")
        self.vectorizer = TfidfVectorizer(max_df=Config.MAX_DF, max_features=1000)
        self.vectors = self.vectorizer.fit_transform(self.corpus)

        if not sparse:
            self.vectors = self.vectors.toarray()


    def transform(self, doc, sparse=True):
        transforemd = self.vectorizer.transform(doc)
        if sparse:
            return transforemd
        else:
            return transforemd.toarray()[0, :]

class word2vec:

    def __init__(self, corpus, idx, dim=50, window=2, training_algorithm='skip', n_epochs=5):
        self.name = 'word2vec'
        self.corpus = corpus
        self.idx = idx

        self.dim = dim
        self.window = window
        self.n_epochs = n_epochs

        logger.info("Inferring word2vec from data")
        self.corpus = [simple_preprocess(doc, deacc=True) for doc in corpus]
        self.vectorizer = W2VTransformer(size=dim,
                                         window=window,
                                         sg=0 if training_algorithm == 'skip' else 1,
                                         iter=n_epochs)
        self.vectorizer = self.vectorizer.fit(self.corpus)
        self.vectors = []
        for doc in self.corpus:
            doc_vector = []
            for word in doc:
                try:
                    doc_vector.append(self.vectorizer.transform(word))
                except:
                    continue
            if len(doc_vector) > 0:
                self.vectors.append(np.mean(doc_vector, axis=0))
            else:
                self.vectors.append(np.ones(shape=(1, dim)))
        self.vectors = np.concatenate(self.vectors, axis=0)

    def transform(self, doc):
        return self.vectorizer.transform(doc)

if __name__ == '__main__':
    data = pnd.read_csv(Config.CLUSTERING_DATA_DIR, encoding='latin1', index_col=0)
    all_text = data.values
    all_text = [text for sublist in all_text for text in sublist]
    indices = data.index.values
    tfidf = TfIdf(all_text, indices)
    word2vec(all_text, indices)