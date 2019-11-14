def extract_bigrams(word):
    bigrams = []
    for i, c in enumerate(word[:-1]):
        bigrams.append('{}{}'.format(c, word[i + 1]))

    return bigrams
