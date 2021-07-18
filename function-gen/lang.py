SOS_token = 0
EOS_token = 1

import pandas as pd
from typing import List, Tuple
import numpy as np

from preprocessing.csv_dataset import CSVDataset

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
    def __init__(self, name, split_char):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.split_char = split_char

    def addSentence(self, sentence):
        if self.split_char != '':
            for word in sentence.split(self.split_char):
                self.addWord(word)
        else:
            for word in list(sentence):
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def transform_to_returns(seq):
    diffed = np.diff(seq)
    return np.concatenate(([seq[0]], diffed), axis=0)


def load_data_int_seq(returns = False) -> Tuple[Lang, Lang, List[Tuple[List[int], str]], List[List[int]], List[str]]:
    train_data = pd.read_csv('./data/eqs.csv')
    test_data = pd.read_csv('./data/eqs-test.csv')
    

    y_train = train_data["eqs"]
    X_train = train_data.drop('eqs', axis = 1)
    X_train = X_train[["0", "1", "2", "3", "4", "5", "6", "7", "8"]].to_numpy()


    y_test = test_data["eqs"]
    X_test = test_data.drop('eqs', axis = 1)
    X_test = X_test[["0", "1", "2", "3", "4", "5", "6", "7", "8"]].to_numpy()
    if returns == True:
        X_train = np.apply_along_axis(transform_to_returns, 1, X_train)
        X_test = np.apply_along_axis(transform_to_returns, 1, X_test)

    seq = Lang("seq", ',')
    for row in X_train:
        seq.addSentence(','.join([str(item) for item in row]))
    for row in X_test:
        seq.addSentence(','.join([str(item) for item in row]))

    eq = Lang("eq", '')
    for row in y_train:
        eq.addSentence(row)
    for row in y_test:
        eq.addSentence(row)


    # train_data_iterable = CSVDataset('./data/eqs.csv', stringify=True)
    # test_data_iterable = CSVDataset('./data/eqs-test.csv', stringify=True)
    # return eq, seq, train_data_iterable, test_data_iterable

    return eq, seq, list(zip(X_train, y_train)), X_test, y_test



# deprecated, old version
def load_data() -> Tuple[Lang, Lang, List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_data = pd.read_csv('./data/eqs.csv')
    test_data = pd.read_csv('./data/eqs-test.csv')
    
    y_train = train_data["eqs"]
    X_train = train_data.drop('eqs', axis = 1)
    X_train = X_train[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()

    y_test = test_data["eqs"]
    X_test = test_data.drop('eqs', axis = 1)
    X_test = X_test[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()

    seq = Lang("seq", ',')
    for row in X_train:
        seq.addSentence(','.join([str(item) for item in row]))
    for row in X_test:
        seq.addSentence(','.join([str(item) for item in row]))

    eq = Lang("eq", '')
    for row in y_train:
        eq.addSentence(row)
    for row in y_test:
        eq.addSentence(row)

    X_train_str = [','.join([str(item) for item in row]) for row in X_train]
    X_test_str = [','.join([str(item) for item in row]) for row in X_train]

    return eq, seq, list(zip(X_train_str, y_train)), list(zip(X_test_str, y_test))


