SOS_token = 0
EOS_token = 1

import pandas as pd

from typing import List

def int_to_binary_str(num: int, width: int) -> str:
    if num >= 0:
        binary = bin(num)[2:].zfill(width)
        return "0" + binary 
    else:
        binary = bin(abs(num))[2:].zfill(width)
        return "1" + binary

def binary_str_to_int(str: str, width: int) -> int:
    sign = str[0]
    num = str[1:]
    num = int(num, 2)

    if sign == "0":
        return num
    else:
        return num *- 1

bin_syms = list('01')
# char to index and index to char maps
bin_char_to_ix = { ch:i for i,ch in enumerate(bin_syms) }
bin_ix_to_char = { i:ch for i,ch in enumerate(bin_syms) }


def bin_encoder(binar: str) -> List[int]:
    # convert data from chars to indices
    output = list(binar)
    for i, ch in enumerate(binar):
        output[i] = bin_char_to_ix[ch]
    return output

def bin_decoder(encoded: List[int]) -> str:
    # convert data from chars to indices
    output = encoded.copy()
    for i, ch in enumerate(encoded):
        output[i] = bin_ix_to_char[ch]
    return ''.join(output)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def load_data():
    data = pd.read_csv('./data/eqs.csv')
    y = data["eqs"]
    X = data.drop('eqs', axis = 1)
    X = X[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
    seq = Lang("seq")
    for row in X:
        seq.addSentence(''.join([str(item) for item in row]))

    eq = Lang("eq")
    for row in y:
        eq.addSentence(row)

    X_str = [','.join([str(item) for item in row]) for row in X]
    return eq, seq, list(zip(X_str, y))

# class IntSeq:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS"}
#         self.n_words = 2  # Count SOS and EOS

#     def addSeq(self, seq):
#         [torch.Tensor(bin_encoder(int_to_binary_str(x, 6))) for x in seq]


# IntSeq("aawd")