import logging

import numpy as np
import pandas as pnd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from Code.Utils.Config import Config
from Code.Utils.preprocess import EnglishTextPreProcessor
from Code.Phase2.tf_idf_encoding import TfIdf

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("KNN")


def most_common(lst):
    return max(set(lst), key=lst.count)


class KNNClassifier:
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

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_docs, True)
        self.validation_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_docs, True)

    def report_on_validation(self):
        self.evaluate_validation_loss_for_ks([1, 5, 9])

    def predict(self, doc, k):
        doc_vector = self.tfidf.get_tfidf_vector(doc, True)
        distances = pairwise_distances(self.train_docs_matrix, doc_vector)[:, 0]
        nearest_docs = np.argsort(distances)[:k]
        nearest_docs_classes = self.train_labels[nearest_docs].tolist()
        return most_common(nearest_docs_classes)

    def evaluate_validation_loss_for_ks(self, ks):
        distances = pairwise_distances(self.validation_docs_matrix, self.train_docs_matrix)
        for k in ks:
            predictions = []
            true_labels = []
            for i, d in enumerate(distances):
                nearest_docs = np.argsort(d)[:k]
                nearest_docs_classes = self.train_labels[nearest_docs].tolist()
                predicted_class = most_common(nearest_docs_classes)
                true_class = self.validation_labels[i]
                predictions.append(predicted_class)
                true_labels.append(true_class)

            accuracy = accuracy_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions, average=None)
            precision = precision_score(true_labels, predictions, average=None)
            f1 = f1_score(true_labels, predictions, average='micro')
            logger.info("K = {}:\n"
                        "\tAccuracy = {}\n"
                        "\tRecall for classes = {}\n"
                        "\tPrecision for classes= {}\n"
                        "\tF1 = {}\n On Validation Data".format(k, accuracy, recall, precision, f1))


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    training_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)

    tfidf = TfIdf("English", training_data, english_text_preprocessor)

    knn_clf = KNNClassifier(training_data, tfidf)
    a = knn_clf.predict("Italy", 5)
    knn_clf.report_on_validation()
    print(a)
