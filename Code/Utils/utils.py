from numpy import uint8
from sys import getsizeof

def extract_bigrams(word):
    bigrams = []
    for i, c in enumerate(word[:-1]):
        bigrams.append('{}{}'.format(c, word[i + 1]))

    return bigrams

def number_to_variable_length(gap):
    res = []

    if gap == 0:
        return [uint8(128)]

    while gap:
        mod = gap % 128
        res.append(uint8(mod))
        gap //= 128

    res[0] += 128
    res.reverse()
    return res

def variable_length_to_posting(variable_length):
    num = 0
    posting = [0]
    for byte in variable_length:
        num *= 128
        if byte >= 128:
            num += byte - 128
            posting.append(num + posting[-1])
            num = 0
        else:
            num += byte
    return posting[1:]

def get_size_list(lst):
    size = 0
    for item in lst:
        size += getsizeof(item)
    return size

def get_size_dict_of_list(dictionary):
    size = 0
    for key, value in dictionary.items():
        size += getsizeof(key) + get_size_list(value)
    return size