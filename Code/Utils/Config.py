class Config:
    DATA_DIR = "Data/"
    PERSIAN_DATA_DIR = DATA_DIR + "Phase1/Persian.csv"
    ENGLISH_DATA_DIR = DATA_DIR + "Phase1/English.csv"
    CACHE_DIR = DATA_DIR + "/Cache/"

    ENGLISH_TRAINING_DATA_DIR = DATA_DIR + "Phase2/phase2_train.csv"
    ENGLISH_TEST_DATA_DIR = DATA_DIR + "Phase2/phase2_test.csv"

    MAX_TF_IDF_FEATURES = 5000
    MAX_DF = 0.85

    TRAINING_DATA_RATIO = 0.8