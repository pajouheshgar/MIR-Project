import Code.Utils.wiki_dump_parser as parser
import pandas as pnd

if __name__ == '__main__':
    sep = ','
    persian_wiki_xml_data_dir = "Data/Phase 1/Persian.xml"
    parser.xml_to_csv(persian_wiki_xml_data_dir, sep)
    persian_wiki_csv_data_dir = "Data/Phase 1/Persian.csv"
    df = pnd.read_csv(persian_wiki_csv_data_dir)
    print(len(df), sep=sep)
    print(df['page_title'][0])
    print(df['text'][0][:100])


