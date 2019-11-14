import logging
import pandas as pnd

import Code.Utils.wiki_dump_parser as parser

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Xml2Csv")

if __name__ == '__main__':
    sep = ','
    persian_wiki_xml_data_dir = "Data/Phase 1/Persian.xml"
    parser.xml_to_csv(persian_wiki_xml_data_dir, sep)
    persian_wiki_csv_data_dir = "Data/Phase 1/Persian.csv"
    df = pnd.read_csv(persian_wiki_csv_data_dir)
    df = df[['page_title', 'text']]
    df.columns = ["Title", "Text"]
    logger.info("Found {} persian documents".format(len(df)))
    df.to_csv("Data/Phase 1/Persian.csv", header=True, index=False)
