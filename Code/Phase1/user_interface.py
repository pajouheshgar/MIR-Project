from Code.Utils.preprocess import EnglishTextPreProcessor, PersianTextPreProcessor
import pandas as pnd
from Code.Phase1.search_engine import SearchEngine, Config


class UI:
    def __init__(self, search_engine):
        self.se = search_engine
        self.level_one_details = "1- Preprocess\n2- Indexing\n" \
                                 "3- Compression\n4- Correction\n" \
                                 "5- Retrieval\n6 - Exit"

        self.level_two_one_details = "1- process input\n2- Show frequent words"
        self.level_two_two_details = "1- Show postings\n2- Show positions"

    def run_level_one(self, command):
        if command == '1':
            flag = False
            while not flag:
                print(self.level_two_one_details)
                c = input()
                if c == '1':
                    print("Enter your text:")
                    text = input()
                    clean_text = self.se.preprocessor.clean_text(text, self.se.stopwords)
                    print(clean_text)
                    flag = True

                elif c == '2':
                    print("Most repeated words are:")
                    print(self.se.stopwords)
                    flag = True
                else:
                    print("Invalid Command")

        elif command == '2':
            flag = False
            while not flag:
                print(self.level_two_two_details)
                c = input()
                if c == '1':
                    print("Enter your word:")
                    word = input()
                    posting = self.se.get_vocab_posting(word)
                    if posting is not None:
                        print(posting)
                    flag = True

                elif c == '2':
                    print("Enter your word:")
                    word = input()
                    posting = self.se.get_vocab_posting(word)
                    positions = self.se.get_vocab_positions(word)
                    print("Word positions in the documents are:")
                    if posting is not None:
                        for p1, p2 in zip(posting, positions):
                            print("Document {}: Positions: {}".format(p1, p2))
                    flag = True
                else:
                    print("Invalid Command")

            pass
        elif command == '3':
            print("Size of postings before compression {}".format(self.se.posting_size))
            print(
                "Size of postings after variable-length compression {}".format(self.se.variable_length_compressed_size))
            print("Size of postings after gamma compression {}".format(self.se.gamma_compressed_size))
        elif command == '4':
            print("Enter your query:")
            query = input()
            corrected_query = self.se.query_spell_correction(query)
            print("Corrected query is:")
            print(corrected_query)

        elif command == '5':
            print("Enter your query:")
            query = input()
            retrieved_documents = self.se.query_lnc_ltc(query)
            print("Retrieved documents by rank:")
            print(retrieved_documents)

        elif command == '6':
            exit()
        else:
            print(self.level_one_details)

    def start(self):
        while True:
            print(self.level_one_details)
            command = input()
            self.run_level_one(command)


if __name__ == '__main__':
    # english_text_preprocessor = EnglishTextPreProcessor()
    # dataframe = pnd.read_csv(Config.ENGLISH_DATA_DIR)
    # search_engine = SearchEngine('English', dataframe, english_text_preprocessor, True)

    persian_text_preprocessor = PersianTextPreProcessor()
    dataframe = pnd.read_csv(Config.PERSIAN_DATA_DIR)
    search_engine = SearchEngine('Persian', dataframe, persian_text_preprocessor, False)

    ui = UI(search_engine)
    ui.start()
