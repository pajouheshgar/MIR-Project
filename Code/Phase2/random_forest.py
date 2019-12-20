import logging

import numpy as np
import pandas as pnd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC

from Code.Phase2.tf_idf_encoding import TfIdf
from Code.Utils.Config import Config
from Code.Utils.preprocess import EnglishTextPreProcessor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Random Forest")


class RandomForestClassifier:
    def __init__(self, data, test_data, tfidf):
        self.data = data
        self.test_data = test_data
        self.tfidf = tfidf

        self.n = len(self.data)
        self.n_train = int(self.n * Config.TRAINING_DATA_RATIO)
        self.train_data = self.data[:self.n_train]

        self.train_docs = self.train_data['Text']
        self.test_docs = self.test_data['Text']
        self.train_labels = self.train_data['Tag'].values
        self.test_labels = self.test_data['Tag'].values

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_docs, False)
        self.test_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.test_docs, False)

        self.model = RFC(max_features=10)
        self.model.fit(self.train_docs_matrix, self.train_labels)

        self.report()

    def predict(self, doc):
        doc_vector = self.tfidf.get_tfidf_vector(doc, sparse=False)
        return self.model.predict([doc_vector])[0]

    def report(self):
        self.evaluate(self.train_docs_matrix, self.train_labels, 'Train')
        self.evaluate(self.test_docs_matrix, self.test_labels, 'Test')

    def evaluate(self, docs_matrix, labels, set_name):
        model = RFC(max_features=10)
        model.fit(self.train_docs_matrix, self.train_labels)
        predictions = model.predict(docs_matrix)
        true_labels = labels

        accuracy = accuracy_score(true_labels, predictions)
        per_class_recall = recall_score(true_labels, predictions, average=None)
        per_class_precision = precision_score(true_labels, predictions, average=None)
        micro_f1 = f1_score(true_labels, predictions, average='micro')
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        np.set_printoptions(precision=3)
        logger.info(" On " + set_name + " Data")
        logger.info("\tAccuracy = {:.3f}\n"
                    "\tRecall_per class = {}\n"
                    "\tPrecision_per class = {}\n"
                    "\tmicro_F1 = {:.3f}\n"
                    "\tmacro_F1 = {:.3f} On Validation Data".format(accuracy, per_class_recall, per_class_precision,
                                                                    micro_f1, macro_f1))


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    training_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)
    testing_data = pnd.read_csv(Config.ENGLISH_TEST_DATA_DIR)

    tfidf = TfIdf("English", training_data, english_text_preprocessor)

    rf_clf = RandomForestClassifier(training_data, testing_data, tfidf)
    # rf_clf.report()

    # print(rf_clf.predict("Italy-"))
