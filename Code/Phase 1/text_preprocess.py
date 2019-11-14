from hazm import Normalizer, sent_tokenize, word_tokenize, stopwords_list, Stemmer
import pandas as pnd

from Code.Utils.clean_persian_text import PersianTextCleaner

persian_text_cleaner = PersianTextCleaner({})
stop_words = set(stopwords_list())
stemmer = Stemmer()
normalizer = Normalizer()


def clean_persian_text(text):
    global normalizer, stemmer
    text = persian_text_cleaner.get_sentences(text)
    normalized_text = normalizer.normalize(text)
    words = word_tokenize(normalized_text)
    words = [w for w in words if w != '.']
    words = [w for w in words if w not in stop_words]
    # words = ["{}:{}".format(w, stemmer.stem(w)) for w in words]
    words = [stemmer.stem(w) for w in words]
    return words




if __name__ == '__main__':
    persian_wiki_csv_data_dir = "Data/Phase 1/Persian.csv"
    df = pnd.read_csv(persian_wiki_csv_data_dir)
    text = df['text'][100]

    print(text)
    print("_" * 100)
    text = clean_persian_text(text)
    print(" ".join(text))
