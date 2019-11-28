import numpy as np

def extract_bigrams(word):
    bigrams = []
    for i, c in enumerate(word[:-1]):
        bigrams.append('{}{}'.format(c, word[i + 1]))

    return bigrams

def number_to_gamma(gap):
    res = []
    while gap:
        res.append(np.bool(gap % 2))
        gap //= 2
    res = res[:-1]
    n = len(res)
    res.append(np.bool(0))
    res += [np.bool(1) for _ in range(n)]
    res.reverse()
    return res

def gamma_to_posting(gamma):
    on_length = True
    length = 0
    num = 0
    posting = [0]
    for bit in gamma:
        if on_length:
            if length == 0 and not bit:
                posting.append(1 + posting[-1])
                continue
            length += 1
            if not bit:
                on_length = False
                num = 1
                length -= 1
        else:
            if length > 0:
                length -= 1
                num *= 2
                num += bit
            if length == 0:
                posting.append(num + posting[-1])
                num = 0
                on_length = True
    return posting[1:]

def number_to_variable_length(gap):
    res = []

    if gap == 0:
        return [np.uint8(128)]

    last = 0
    while gap:
        mod = gap % 128
        if last:
            res.append(np.uint8(mod))
        else:
            res.append(np.uint8(mod + 128))
            last = 1
        gap //= 128

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

def get_size_dict_of_list(dictionary):
    size = 0
    size_map = {np.bool: 1./8., np.uint8: 1, int: 8}
    for key, value in dictionary.items():
        size += len(value) * size_map[type(value[0])]
    return int(np.ceil(size))

def edit_distance(str1, str2):
    n1 = len(str1)
    n2 = len(str2)
    dp = np.zeros(shape=(n1 + 1, n2 + 1), dtype=int)
    for i in range(1, n1 + 1):
        dp[i, 0] = i
    for j in range(1, n2 + 1):
        dp[0, j] = j
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            dp[i, j] = min(dp[i, j - 1], dp[i - 1, j]) + 1
            if str1[i - 1] == str2[j - 1]:
                dp[i, j] = min(dp[i, j], dp[i - 1, j - 1])
            else:
                dp[i, j] = min(dp[i, j], dp[i - 1, j - 1] + 2)
    return dp[n1, n2]