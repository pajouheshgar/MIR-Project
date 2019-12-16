import pickle
import logging

import pandas as pnd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from Code.Utils.Config import Config
from Code.Utils.preprocess import EnglishTextPreProcessor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("tf-idf")


class TfIdf:
    def __init__(self, name, training_data, text_preprocessor, clear_cache=False):
        self.name = name
        self.train_data = training_data
        self.text_preprocessor = text_preprocessor

        self.cache_dir = Config.CACHE_DIR + self.name + "/"
        with open(self.cache_dir + "stopwords", "rb") as f:
            logger.info("Loading stopwords from a file")
            stopwords = pickle.load(f)

        logger.info("Inferring tf-idf from training data")
        self.cv = CountVectorizer(max_df=Config.MAX_DF, stop_words=stopwords, max_features=Config.MAX_TF_IDF_FEATURES)
        docs = training_data['Text']
        docs = [self.text_preprocessor.pre_stopword_text_cleaning(doc) for doc in docs]
        word_count_vector = self.cv.fit_transform(docs)
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.tfidf_transformer.fit(word_count_vector)
        self.tfidf_transformer.fit_transform(word_count_vector)

    def get_tfidf_vector(self, doc, sparse=True):
        doc = [self.text_preprocessor.pre_stopword_text_cleaning(doc)]
        word_tfidf_pair = self.tfidf_transformer.transform(self.cv.transform(doc))
        if sparse:
            return word_tfidf_pair
        else:
            return word_tfidf_pair.toarray()[0, :]

    def get_tf_idf_vector_for_docs(self, docs, sparse=True):
        word_tfidf_pair = self.tfidf_transformer.transform(self.cv.transform(docs))
        if sparse:
            return word_tfidf_pair
        else:
            return word_tfidf_pair.toarray()


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    train_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)

    tfidf = TfIdf("English", train_data, english_text_preprocessor)

    a = tfidf.get_tfidf_vector("hello usa italy said do done", False)
    print(a)
