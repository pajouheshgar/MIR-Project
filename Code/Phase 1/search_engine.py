import os
import pickle
import logging
from collections import defaultdict

from tqdm import tqdm
import pandas as pnd

from Code.Utils.preprocess import PersianTextPreProcessor, EnglishTextPreProcessor
from Code.Utils.utils import *

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Search Engine")


class Config:
    DATA_DIR = "Data/"
    PERSIAN_DATA_DIR = DATA_DIR + "Phase 1/Persian.csv"
    ENGLISH_DATA_DIR = DATA_DIR + "Phase 1/English.csv"
    CACHE_DIR = DATA_DIR + "/Cache/"


class SearchEngine:

    def __init__(self, name, df, preprocessor, clear_cache=True):
        self.name = name
        self.dataframe = df
        self.titles = self.dataframe['Title']
        self.documents = self.dataframe['Text']
        self.n_documents = len(self.documents)
        self.document_words = []
        self.preprocessor = preprocessor
        self.cache_dir = Config.CACHE_DIR + name + "/"
        try:
            os.mkdir(self.cache_dir)
        except FileExistsError:
            pass

        if clear_cache:
            self.vocab_frequency = defaultdict(lambda: 1)
            self.stopwords = self.infer_stopwords()
            logger.info("Inferred stopwords are: {}".format(self.stopwords))
            self.print_most_repeated_words(50)

            self.document_words = list(
                map(lambda words: self.preprocessor.remove_stop_words_and_stem(words, self.stopwords),
                    self.document_words)
            )

            with open(self.cache_dir + "vocab_frequency", "wb") as f:
                logger.info("Saving vocab_frequency into a file")
                pickle.dump(dict(self.vocab_frequency), f)

            with open(self.cache_dir + "stopwords", "wb") as f:
                logger.info("Saving stopwords into a file")
                pickle.dump(self.stopwords, f)

            with open(self.cache_dir + "document_words", "wb") as f:
                logger.info("Saving document_words into a file")
                pickle.dump(self.document_words, f)

            self.postings = defaultdict(lambda: [])
            self.variable_length_compressed = defaultdict(lambda: [])
            self.positional_index = defaultdict(lambda: [])
            self.bigram_index = defaultdict(lambda: set())
            self.tfdf = defaultdict(lambda : [0, 0])
            self.build_index()
            self.variable_length_compression()

            with open(self.cache_dir + "postings", "wb") as f:
                logger.info("Saving postings into a file")
                pickle.dump(dict(self.postings), f)

            with open(self.cache_dir + "positional_index", "wb") as f:
                logger.info("Saving positional_index into a file")
                pickle.dump(dict(self.positional_index), f)

            with open(self.cache_dir + "bigram_index", "wb") as f:
                logger.info("Saving bigram_index into a file")
                pickle.dump(dict(self.bigram_index), f)

            with open(self.cache_dir + "tfdf", "wb") as f:
                logger.info("Saving tfdf into a file")
                pickle.dump(dict(self.tfdf), f)

            with open(self.cache_dir + "variable_length_compressed", "wb") as f:
                logger.info("Saving postings into a file")
                pickle.dump(dict(self.variable_length_compressed), f)


        else:
            with open(self.cache_dir + "vocab_frequency", "rb") as f:
                logger.info("Loading vocab_frequency from file")
                self.vocab_frequency = pickle.load(f)

            with open(self.cache_dir + "stopwords", "rb") as f:
                logger.info("Loading stopwords from file")
                self.stopwords = pickle.load(f)

            with open(self.cache_dir + "document_words", "rb") as f:
                logger.info("Loading document_words from file")
                self.document_words = pickle.load(f)

            with open(self.cache_dir + "postings", "rb") as f:
                logger.info("Loading postings from file")
                self.postings = pickle.load(f)

            with open(self.cache_dir + "positional_index", "rb") as f:
                logger.info("Loading positional_index from file")
                self.positional_index = pickle.load(f)

            with open(self.cache_dir + "bigram_index", "rb") as f:
                logger.info("Loading bigram_index from file")
                self.bigram_index = pickle.load(f)

            with open(self.cache_dir + "tfdf", "rb") as f:
                logger.info("Loading tfdf from file")
                self.tfdf = pickle.load(f)

            with open(self.cache_dir + "variable_length_compressed", "wb") as f:
                logger.info("Loading compressed postings from file")
                self.variable_length_compressed = pickle.load(f)

        posting_size = get_size_dict_of_list(self.postings)
        variable_length_compressed_size = get_size_dict_of_list(self.variable_length_compressed)

        logger.info("Size of postings before compression {}".format(posting_size))
        logger.info("Size of postings after variable-length compression {}".format(variable_length_compressed_size))

    def infer_stopwords(self):
        logger.info("Inferring stopwords from documents")
        for title, document in zip(self.titles, tqdm(self.documents, position=0, leave=True)):
            document_words = self.preprocessor.pre_stopword_process(document)
            n = len(document_words)
            if n == 0:
                continue
            f = 1 / n
            for w in document_words:
                self.vocab_frequency[w] += f
            self.document_words.append(document_words)

        sorted_words = sorted(self.vocab_frequency.items(), key=lambda x: (x[1] * (len(x[0]) < 4)), reverse=True)
        logger.info("Found {} words".format(len(sorted_words)))
        stopwords = set(
            map(
                lambda x: x[0],
                filter(lambda x: x[1] > (2.9 * self.n_documents / 1572), sorted_words[:50]))
        )
        return stopwords

    def print_most_repeated_words(self, top_k):
        sorted_words = sorted(self.vocab_frequency.items(), key=lambda x: x[1], reverse=True)
        top_k_words = [w[0] for w in sorted_words[:top_k]]
        logger.info("Top {} words are: {}".format(top_k, top_k_words))

    def build_index(self):
        logger.info("Building index from documents...")
        for i, doc_words in enumerate(tqdm(self.document_words, position=0, leave=True)):
            for j, word in enumerate(doc_words):
                self.tfdf[word][0] += 1
                if len(self.postings[word]) == 0:
                    self.postings[word].append(i)
                    self.positional_index[word].append([])
                    self.positional_index[word][-1].append(j)
                    self.tfdf[word][1] += 1
                elif self.postings[word][-1] != i:
                    self.postings[word].append(i)
                    self.positional_index[word].append([])
                    self.positional_index[word][-1].append(j)
                    self.tfdf[word][1] += 1
                else:
                    self.positional_index[word][-1].append(j)

                bigrams = extract_bigrams(word)
                for bigram in bigrams:
                    self.bigram_index[bigram].add(word)

    def variable_length_compression(self):
        for word, posting in self.postings.items():
            last = 0
            for id in posting:
                self.variable_length_compressed[word] += number_to_variable_length(id - last)
                last = id

    def variable_length_decompress(self):
        postings = defaultdict(lambda: [])
        for word, posting in self.variable_length_compressed.items():
            postings[word] = variable_length_to_posting(posting)
        return postings

    def get_vocab_posting(self, vocab):
        vocab = self.preprocessor.process_single_word(vocab)
        if vocab in self.postings:
            return self.postings[vocab]
        else:
            print("Vocab not found in the dictionary")

    def get_vocab_positions(self, vocab):
        vocab = self.preprocessor.process_single_word(vocab)
        if vocab in self.positional_index:
            return self.positional_index[vocab]
        else:
            print("Vocab not found in the dictionary")


if __name__ == '__main__':
    # persian_text_preprocessor = PersianTextPreProcessor()
    # dataframe = pnd.read_csv(Config.PERSIAN_DATA_DIR)
    # search_engine = SearchEngine('Persian', dataframe, persian_text_preprocessor, True)
    #
    # print(search_engine.get_vocab_posting('ایران'))
    # print(search_engine.get_vocab_positions('ایران'))

    english_text_preprocessor = EnglishTextPreProcessor()
    dataframe = pnd.read_csv(Config.ENGLISH_DATA_DIR)
    search_engine = SearchEngine('English', dataframe, english_text_preprocessor, True)

    print(search_engine.get_vocab_posting('News'))
    print(search_engine.get_vocab_positions('News'))
