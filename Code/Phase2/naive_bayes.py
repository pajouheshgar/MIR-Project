import logging

import numpy as np
import pandas as pnd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from Code.Phase2.tf_idf_encoding import TfIdf
from Code.Utils.Config import Config
from Code.Utils.preprocess import EnglishTextPreProcessor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("NaiveBayes")


class NaiveBayesClassifier:
    def __init__(self, data, test_data, tfidf):
        self.data = data
        self.test_data = test_data
        self.tfidf = tfidf

        self.n = len(self.data)
        self.n_train = int(self.n * Config.TRAINING_DATA_RATIO)
        self.train_data = self.data

        self.train_docs = self.train_data['Text']
        self.test_docs = self.test_data['Text']
        self.train_labels = self.train_data['Tag'].values
        self.test_labels = self.test_data['Tag'].values

        self.class_priors = [np.sum(self.train_labels == i) / self.n_train for i in range(1, 5)]

        self.vocab2id = self.tfidf.cv.vocabulary_
        self.vocab_class_cooccurrences = [[0, 0, 0, 0] for _ in range(len(self.vocab2id))]
        for class_id, doc in zip(self.train_labels, self.train_docs):
            for vocab in self.tfidf.text_preprocessor.pre_stopword_process(doc):
                if vocab not in self.vocab2id:
                    continue
                self.vocab_class_cooccurrences[self.vocab2id[vocab]][class_id - 1] += 1

        self.class_total_vocabs = [0, 0, 0, 0]
        for c in self.vocab_class_cooccurrences:
            for i, x in enumerate(c):
                self.class_total_vocabs[i] += x

        self.vocab_conditional_probabilities = []
        for vco in self.vocab_class_cooccurrences:
            self.vocab_conditional_probabilities.append([
                vco[0] / self.class_total_vocabs[0],
                vco[1] / self.class_total_vocabs[1],
                vco[2] / self.class_total_vocabs[2],
                vco[3] / self.class_total_vocabs[3],
            ])

        self.report()

    def predict(self, doc):
        class_scores = [np.log(p) for p in self.class_priors]
        for c in range(4):
            for vocab in self.tfidf.text_preprocessor.pre_stopword_process(doc):
                if vocab not in self.vocab2id:
                    continue
                class_scores[c] += np.log(self.vocab_conditional_probabilities[self.vocab2id[vocab]][c])

        return np.argmax(class_scores) + 1

    def report(self):
        self.evaluate(self.train_docs, self.train_labels, set_name='Train')
        self.evaluate(self.test_docs, self.test_labels, set_name='Test')

    def evaluate(self, docs, labels, set_name):
        predictions = []
        true_labels = []
        for doc, label in zip(docs, labels):
            predictions.append(self.predict(doc))
            true_labels.append(label)

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
                    "\tmacro_F1 = {:.3f}".format(accuracy, per_class_recall, per_class_precision, micro_f1, macro_f1))

if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    training_data = pnd.read_csv(Config.ENGLISH_TRAINING_DATA_DIR)
    testing_data = pnd.read_csv(Config.ENGLISH_TEST_DATA_DIR)

    tfidf = TfIdf("English", training_data, english_text_preprocessor)

    nb_clf = NaiveBayesClassifier(training_data, testing_data, tfidf)
    # print(nb_clf.predict("Italy-"))
