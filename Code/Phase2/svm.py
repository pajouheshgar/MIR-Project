import logging

import numpy as np
import pandas as pnd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import LinearSVC

from Code.Phase2.tf_idf_encoding import TfIdf
from Code.Utils.Config import Config
from Code.Utils.preprocess import EnglishTextPreProcessor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("NaiveBayes")


class SVMClassifier:
    def __init__(self, data, tfidf, C):
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

        self.model = LinearSVC(C=C)
        self.model.fit(self.train_docs_matrix, self.train_labels)

    def predict(self, doc):
        doc_vector = self.tfidf.get_tfidf_vector(doc, sparse=False)
        return self.model.predict([doc_vector])[0]

    def report_on_validation(self):
        for C in [0.5, 1.0, 1.5, 2.0]:
            model = LinearSVC(C=C)
            model.fit(self.train_docs_matrix, self.train_labels)
            predictions = model.predict(self.validation_docs_matrix)
            true_labels = self.validation_labels

            accuracy = accuracy_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions, average=None)
            precision = precision_score(true_labels, predictions, average=None)
            f1 = f1_score(true_labels, predictions, average='micro')
            logger.info("C = {}:\n"
                        "\tAccuracy = {}\n"
                        "\tRecall for classes = {}\n"
                        "\tPrecision for classes= {}\n"
                        "\tF1 = {}\n On Validation Data".format(C, accuracy, recall, precision, f1))


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    training_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)

    tfidf = TfIdf("English", training_data, english_text_preprocessor)

    svm_clf = SVMClassifier(training_data, tfidf, 0.5)
    svm_clf.report_on_validation()

    print(svm_clf.predict("Italy-"))
