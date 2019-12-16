import logging

import numpy as np
import pandas as pnd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC

from Code.Phase2.tf_idf_encoding import TfIdf
from Code.Utils.Config import Config
from Code.Utils.preprocess import EnglishTextPreProcessor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("NaiveBayes")


class RandomForestClassifier:
    def __init__(self, data, tfidf):
        self.data = data
        self.tfidf = tfidf

        self.n = len(self.data)
        self.n_train = int(self.n * Config.TRAINING_DATA_RATIO)
        self.train_data = self.data[:self.n_train]
        self.validation_data = self.data[self.n_train:]

        self.train_docs = self.train_data['Text']
        self.validation_docs = self.validation_data['Text']
        self.train_labels = self.train_data['Tag'].values
        self.validation_labels = self.validation_data['Tag'].values

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_docs, False)
        self.validation_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_docs, False)

        self.model = RFC(max_features=10)
        self.model.fit(self.train_docs_matrix, self.train_labels)

    def predict(self, doc):
        doc_vector = self.tfidf.get_tfidf_vector(doc, sparse=False)
        return self.model.predict([doc_vector])[0]

    def report_on_validation(self):
        model = RFC(max_features=10)
        model.fit(self.train_docs_matrix, self.train_labels)
        predictions = model.predict(self.validation_docs_matrix)
        true_labels = self.validation_labels

        accuracy = accuracy_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions, average=None)
        precision = precision_score(true_labels, predictions, average=None)
        f1 = f1_score(true_labels, predictions, average='micro')
        logger.info("RandomForest result on validation data\n"
                    "\tAccuracy = {}\n"
                    "\tRecall for classes = {}\n"
                    "\tPrecision for classes= {}\n"
                    "\tF1 = {}".format(accuracy, recall, precision, f1))


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    training_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)

    tfidf = TfIdf("English", training_data, english_text_preprocessor)

    rf_clf = RandomForestClassifier(training_data, tfidf)
    rf_clf.report_on_validation()

    print(rf_clf.predict("Italy-"))
