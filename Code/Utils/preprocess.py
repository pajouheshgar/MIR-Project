import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from hazm import Normalizer, word_tokenize, Stemmer

from Code.Utils.clean_persian_text import PersianTextCleaner


class PersianTextPreProcessor:
    def __init__(self):
        self.stemmer = Stemmer()
        self.normalizer = Normalizer()
        self.punctuations = string.punctuation

    def process_single_word(self, word):
        word = word.lower()
        word = re.sub('\d+', '', word)
        word = word.translate(str.maketrans(self.punctuations, ' ' * len(self.punctuations)))
        word = ' '.join(re.sub(r'[^ضصثقفغعهخحجچشسیبلاتنمکگظطزرذدپوئژآؤ \n]', ' ', word).split())
        word = word.strip()
        word = self.normalizer.normalize(word)
        word = self.stemmer.stem(word)
        return word

    def pre_stopword_process(self, text):
        # text = self.persian_text_cleaner.get_sentences(text)
        text = text.lower()
        text = re.sub('\d+', '', text)
        text = text.translate(str.maketrans(self.punctuations, ' ' * len(self.punctuations)))
        text = ' '.join(re.sub(r'[^ضصثقفغعهخحجچشسیبلاتنمکگظطزرذدپوئژآؤ \n]', ' ', text).split())
        text = text.strip()
        normalized_text = self.normalizer.normalize(text)
        words = word_tokenize(normalized_text)
        words = [w for w in words if w != '.']
        return words

    def clean_text(self, text, stopwords, remove_stopwords=True, stem=True):
        words = self.pre_stopword_process(text)
        if remove_stopwords:
            words = [w for w in words if w not in stopwords]

        if stem:
            words = [self.stemmer.stem(w) for w in words]
        return words

    def stem(self, words):
        words = [self.stemmer.stem(w) for w in words]
        return words


class EnglishTextPreProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.punctuations = string.punctuation

    def process_single_word(self, word):
        word = word.lower()
        word = re.sub('\d+', ' ', word)
        # word = word.translate(str.maketrans(' ', ' ', string.punctuation))
        word = word.translate(str.maketrans(self.punctuations, ' ' * len(self.punctuations)))
        word = word.strip()
        word = self.stemmer.stem(word)
        return word

    def pre_stopword_process(self, text):
        text = text.lower()
        text = re.sub('\d+', ' ', text)
        # text = text.translate(str.maketrans(' ', ' ', string.punctuation))
        text = text.translate(str.maketrans(self.punctuations, ' ' * len(self.punctuations)))

        text = text.strip()
        words = word_tokenize(text)
        return words

    def clean_text(self, text, stopwords, remove_stopwords=True, stem=True):
        words = self.pre_stopword_process(text)
        if remove_stopwords:
            words = [w for w in words if w not in stopwords]
        if stem:
            words = [self.stemmer.stem(w) for w in words]
        return words

    def stem(self, words):
        words = [self.stemmer.stem(w) for w in words]
        return words
