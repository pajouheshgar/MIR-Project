import logging
from collections import defaultdict

import pandas as pnd

from Code.Utils.preprocess import PersianTextPreProcessor


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Search Engine")


class Config:
    INFER_STOPWORDS = True
    PERSIAN_DATA_DIR = "Data/Phase 1/Persian.csv"
    ENGLISH_DATA_DIR = "Data/Phase 1/English.csv"


class SearchEngine:

    def __init__(self, dataframe, preprocessor):
        self.dataframe = dataframe
        self.titles = self.dataframe['Title']
        self.documents = self.dataframe['Text']
        self.preprocessor = preprocessor

        self.vocab_frequency = defaultdict(lambda: 1)
        self.infer_stopwords()

    def infer_stopwords(self):
        for title, document in zip(self.titles, self.documents):
            document_words = self.preprocessor.pre_stopword_process(document)
            n = len(document_words)
            if n < 10:
                continue

            f = 1 / n
            for w in document_words:
                self.vocab_frequency[w] += f

        sorted_words = sorted(self.vocab_frequency.items(), key=lambda x: x[1], reverse=True)
        logger.info("Found {} words".format(len(sorted_words)))
        print(sorted_words[:20])


if __name__ == '__main__':
    persian_text_preprocessor = PersianTextPreProcessor()
    dataframe = pnd.read_csv(Config.PERSIAN_DATA_DIR)
    search_engine = SearchEngine(dataframe, persian_text_preprocessor)

