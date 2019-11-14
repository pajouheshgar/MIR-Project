from hazm import Normalizer, word_tokenize, stopwords_list, Stemmer

from Code.Utils.clean_persian_text import PersianTextCleaner


class PersianTextPreProcessor:
    def __init__(self):
        self.persian_text_cleaner = PersianTextCleaner({})
        self.stop_words = set(stopwords_list())
        self.stemmer = Stemmer()
        self.normalizer = Normalizer()

    def pre_stopword_process(self, text):
        text = self.persian_text_cleaner.get_sentences(text)
        normalized_text = self.normalizer.normalize(text)
        words = word_tokenize(normalized_text)
        words = [w for w in words if w != '.']
        return words

    def clean_text(self, text):
        words = self.pre_stopword_process(text)
        words = [w for w in words if w not in self.stop_words]
        words = [self.stemmer.stem(w) for w in words]
        return words
