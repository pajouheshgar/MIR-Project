import logging

import numpy as np
import pandas as pnd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import LinearSVC

from Code.Phase2.tf_idf_encoding import TfIdf
from Code.Utils.Config import Config
from Code.Utils.preprocess import EnglishTextPreProcessor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("SVM")


class SVMClassifier:
    def __init__(self, data, test_data, tfidf):
        self.data = data
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

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_docs, False)
        self.validation_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_docs, False)
        self.test_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.test_docs, False)

        self.report()

    def predict(self, doc, C):
        doc_vector = self.tfidf.get_tfidf_vector(doc, sparse=False)
        return self.model.predict([doc_vector])[0]

    def report(self):
        self.model, best_C = self.report_on_validation([0.5, 1.0, 1.5, 2.0])
        self.evaluate(self.train_docs_matrix, self.train_labels, best_C, 'Train', model=self.model)
        self.evaluate(self.test_docs_matrix, self.test_labels, best_C, 'Test', model=self.model)

    def build_model(self, C):
        model = LinearSVC(C=C)
        model.fit(self.train_docs_matrix, self.train_labels)
        return model

    def evaluate(self, docs_matrix, labels, C, set_name, model=None):
        if not model:
            model = self.build_model(C)
        predictions = model.predict(docs_matrix)
        true_labels = labels

        accuracy = accuracy_score(true_labels, predictions)
        per_class_recall = recall_score(true_labels, predictions, average=None)
        per_class_precision = precision_score(true_labels, predictions, average=None)
        micro_f1 = f1_score(true_labels, predictions, average='micro')
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        np.set_printoptions(precision=3)
        logger.info(" On " + set_name + " Data")
        logger.info("C = {}:\n"
                    "\tAccuracy = {:.3f}\n"
                    "\tRecall_per class = {}\n"
                    "\tPrecision_per class = {}\n"
                    "\tmicro_F1 = {:.3f}\n"
                    "\tmacro_F1 = {:.3f}".format(C, accuracy,
                                                 per_class_recall, per_class_precision,
                                                 micro_f1, macro_f1))
        return model, micro_f1

    def report_on_validation(self, Cs):
        micro_f1_list = []
        model_list = []
        for C in Cs:
            model, micro_f1 = self.evaluate(self.validation_docs_matrix, self.validation_labels, C, set_name='Validation')
            micro_f1_list.append(micro_f1)
            model_list.append(model)

        best_idx = np.argmax(micro_f1_list)
        return model_list[best_idx], Cs[best_idx]


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    training_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)
    testing_data = pnd.read_csv(Config.ENGLISH_TEST_DATA_DIR)

    tfidf = TfIdf("English", training_data, english_text_preprocessor)

    svm_clf = SVMClassifier(training_data, testing_data, tfidf)

    # print(svm_clf.predict("Italy-"))
