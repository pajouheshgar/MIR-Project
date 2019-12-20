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
    def __init__(self, train_data, test_data, tfidf):
        self.data = train_data
        self.test_data = test_data
        self.tfidf = tfidf

        self.n = len(self.data)
        self.n_train = int(self.n * Config.TRAINING_DATA_RATIO)
        self.train_data = self.data[:self.n_train]
        self.validation_data = self.data[self.n_train:]

        self.train_docs = self.train_data['Text']
        self.validation_docs = self.validation_data['Text']
        self.test_docs = self.test_data['Text']
        self.train_labels = self.train_data['Tag'].values
        self.validation_labels = self.validation_data['Tag'].values
        self.test_labels = self.test_data['Tag'].values

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_docs, True)
        self.validation_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_docs, True)
        self.test_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.test_docs, True)

        self.report()
    def report(self):
        best_k = self.evaluate_validation_for_ks([1, 5, 9])
        logger.info("Best value for k: {}".format(best_k))
        self.evaluate(self.train_docs_matrix, self.train_labels, best_k, set_name='Train')
        self.evaluate(self.test_docs_matrix, self.test_labels, best_k, set_name='Test')


    def predict(self, doc, k):
        doc_vector = self.tfidf.get_tfidf_vector(doc, True)
        distances = pairwise_distances(self.train_docs_matrix, doc_vector)[:, 0]
        nearest_docs = np.argsort(distances)[:k]
        nearest_docs_classes = self.train_labels[nearest_docs].tolist()
        return most_common(nearest_docs_classes)

    def evaluate(self, docs, labels, k, set_name):
        distances = pairwise_distances(docs, self.train_docs_matrix)
        predictions = []
        true_labels = []
        for i, d in enumerate(distances):
            nearest_docs = np.argsort(d)[:k]
            nearest_docs_classes = self.train_labels[nearest_docs].tolist()
            predicted_class = most_common(nearest_docs_classes)
            true_class = labels[i]
            predictions.append(predicted_class)
            true_labels.append(true_class)

        accuracy = accuracy_score(true_labels, predictions)
        per_class_recall = recall_score(true_labels, predictions, average=None)
        per_class_precision = precision_score(true_labels, predictions, average=None)
        micro_f1 = f1_score(true_labels, predictions, average='micro')
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        np.set_printoptions(precision=3)
        logger.info("On " + set_name + " Data")
        logger.info("K = {}:\n"
                    "\tAccuracy = {:.3f}\n"
                    "\tRecall_per class = {}\n"
                    "\tPrecision_per class = {}\n"
                    "\tmicro_F1 = {:.3f}\n"
                    "\tmacro_F1 = {:.3f}".format(k, accuracy, per_class_recall, per_class_precision,
                                                 micro_f1, macro_f1))
        return micro_f1

    def evaluate_validation_for_ks(self, ks):
        micro_f1_list = []
        for k in ks:
            micro_f1_list.append(self.evaluate(self.validation_docs_matrix, self.validation_labels, k,
                                               set_name="Validation"))
        return ks[np.argmax(micro_f1_list)]



if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    training_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)
    testing_data = pnd.read_csv(Config.ENGLISH_TEST_DATA_DIR)

    tfidf = TfIdf("English", training_data, english_text_preprocessor)

    knn_clf = KNNClassifier(training_data, testing_data, tfidf)
    # a = knn_clf.predict("Italy", 5)
    # print(a)
