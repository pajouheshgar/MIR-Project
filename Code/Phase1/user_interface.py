from Code.Utils.preprocess import EnglishTextPreProcessor, PersianTextPreProcessor
import pandas as pnd
from Code.Phase1.search_engine import SearchEngine, Config


class UI:
    def __init__(self, search_engine):
        self.se = search_engine
        self.level_one_details = "1- Preprocess\n2- Indexing\n" \
                                 "3- Compression\n4- Correction\n" \
                                 "5- Retrieval\n6 - Remove Doc\n" \
                                 "7- Add Doc\n8- Exit"

        self.level_two_one_details = "1- process input\n2- Show frequent words"
        self.level_two_two_details = "1- Show postings\n2- Show positions"
        self.level_two_five_details = "1- Normal search\n2- Proximity search"
        self.level_three_five_details = "1- With Class\n2- Without Class"
        self.level_four_five_details = "1- KNN\n2- Naive Bayes\n3- SVM\n4- Random Forest"
        self.map_classifier = {1: 'KNN', 2: 'Naive Bayes', 3: 'SVM', 4: 'Random Forest'}

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
            while True:
                print(self.level_two_five_details)
                search_type = input()
                if search_type == '1' or search_type == '2':
                    break
                else:
                    print("Invalid Command")

            flag = False
            while not flag:
                print(self.level_three_five_details)
                is_class = input()
                if is_class == '1':
                    while True:
                        print(self.level_four_five_details)
                        classifier = input()
                        if classifier == '1' or classifier == '2' or classifier == '3' or classifier == '4':
                            classifier_name = self.map_classifier[int(classifier)]
                            break
                        else:
                            print("Invalid classifier number")
                    print("Enter your query:")
                    query = input()
                    while True:
                        print("Enter class number (1-World, 2-Sports, 3-Business, 4-Sci/Tech):")
                        class_label = input()
                        if class_label == '1' or class_label == '2' or class_label == '3' or class_label == '4':
                            break
                        else:
                            print("Invalid class number")

                    if search_type == '1':
                        retrieved_documents = self.se.query_lnc_ltc(query, num_class=int(class_label),
                                                                    classifier=classifier_name)
                    else:
                        print("Enter window size:")
                        window_size = int(input())
                        retrieved_documents = self.se.query_lnc_ltc_proximity(query, window_size,
                                                                              num_class=int(class_label),
                                                                              classifier=classifier_name)
                    print("Retrieved documents by rank:")
                    print(retrieved_documents)
                    flag = True

                elif is_class == '2':
                    print("Enter your query:")
                    query = input()
                    if search_type == '1':
                        retrieved_documents = self.se.query_lnc_ltc(query)
                    else:
                        print("Enter window size:")
                        window_size = int(input())
                        retrieved_documents = self.se.query_lnc_ltc_proximity(query, window_size)
                    print("Retrieved documents by rank:")
                    print(retrieved_documents)
                    flag = True
                else:
                    print("Invalid Command")

        elif command == '6':
            print("Enter doc_id to remove")
            doc_id = int(input())
            self.se.remove_doc(doc_id)
        elif command == '7':
            print("Enter document content")
            text = input()
            self.se.add_doc(text)

        elif command == '8':
            exit()
        else:
            print(self.level_one_details)

    def start(self):
        while True:
            print(self.level_one_details)
            command = input()
            self.run_level_one(command)


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreProcessor()
    dataframe = pnd.read_csv(Config.ENGLISH_DATA_DIR)
    search_engine = SearchEngine('English', dataframe, english_text_preprocessor, True)
    #
    # persian_text_preprocessor = PersianTextPreProcessor()
    # dataframe = pnd.read_csv(Config.PERSIAN_DATA_DIR)
    # search_engine = SearchEngine('Persian', dataframe, persian_text_preprocessor, False)

    ui = UI(search_engine)
    ui.start()
