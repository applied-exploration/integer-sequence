from random import choice
from typing import List
import numpy as np
import torch

def flatten(list_of_lists):
    valid_lists = all(isinstance(elem, (list, np.ndarray, tuple)) for elem in list_of_lists)
    
    if valid_lists:
        return [item for sublist in list_of_lists for item in sublist]
    else: return list_of_lists

def generate_random_eq(length: int) -> str:
    def generate_next(prev: str) -> str:
        if prev in ['1','2','3','4','5','6','7','8','9','0']:
            return choice(['+', '-', '*'])
        elif prev in ['s', '+', '-', '*']:
            return choice(['1','2','3','4','5','6','7','8','9','0', 't', 't', 't', 't'])
        elif prev in ['t']:
            return choice(['+', '-', '*'])
        else:
            raise ValueError('Unexpected prev character')

    result: List[str] = []
    for i in range(0, length):
        last_char = generate_next('s') if i == 0 else generate_next(result[i-1])
        result.append(last_char)
    return ''.join(result)

def generate_random_eq_valid(length: int) -> str:
    valid = False
    while valid == False:
        eq = generate_random_eq(length)
        valid = is_eq_valid(eq)
    return eq


def is_eq_valid(eq: str, test_set: List[int] = [1,2,4,5,10]) -> bool:
    try:
        # if "t" not in eq:
        #     return False
        results = [eval(eq.replace('t', str(num))) for num in test_set]
        if len(set(results)) == 1:
            return False
        elif zoo in results:
            return False
        elif nan in results:
            return False
        else:
            return True
    except:
        return False


def eq_to_seq(eq: str, length: int) -> List[int]:
    int_seq: List[int] = []
    for i in range(0, length):
        try:
            # prev_2 = int_seq[i-2] if i > 2 else 0
            # prev_1 = int_seq[i-1] if i > 1 else 1
            # int_seq.append(int(parse_expr(eq, local_dict = {'t': i+1, 'x': prev_1, 'y': prev_2 })))
            
            int_seq.append(int(eval(eq.replace('t', str(i+1)))))
        except:
            pass
    if len(int_seq) != length: return [0] * length
    return int_seq

syms = list('+*-0123456789t')
# char to index and index to char maps
char_to_ix = { ch:i for i,ch in enumerate(syms) }
ix_to_char = { i:ch for i,ch in enumerate(syms) }

def eq_encoder(eq: str) -> List[int]:
    # convert data from chars to indices
    output = list(eq)
    for i, ch in enumerate(eq):
        output[i] = char_to_ix[ch]
    return output

def eq_decoder(encoded: List[int]) -> str:
    # convert data from chars to indices
    output = encoded.copy()
    for i, ch in enumerate(encoded):
        output[i] = ix_to_char[ch]
    return ''.join(output)

def accuracy_score(pred: List[str], target: List[str]) -> float:
    is_valid = [eq_to_seq(pair[0], 9) == eq_to_seq(pair[1], 9) for pair in zip(pred, target)]

    accuracy = is_valid.count(1) / len(is_valid)

    return accuracy

def mae_score(pred: List[str], target: List[str]) -> float:

    def mae(lhs: List[int], rhs: List[int]) -> int:
        difference = [abs(n1 - n2) for n1, n2 in zip(lhs, rhs)]
        return sum(difference)
    maes = [mae(eq_to_seq(pair[0], 9), eq_to_seq(pair[1], 9)) for pair in zip(pred, target)]

    return sum(maes) / len(pred)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))




def normalize_0_1(a: np.ndarray) -> np.ndarray:
    # Normalised [0,1]
    return (a - np.min(a))/np.ptp(a)

def normalize_1_255(a: np.ndarray) -> np.ndarray:
    # Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
    return (255*(a - np.min(a))/np.ptp(a)).astype(int)

def normalize_minus1_1(a: np.ndarray) -> np.ndarray:
    # Normalised [-1,1]
    return 2.*(a - np.min(a))/np.ptp(a)-1


def remove_key(d, key):
    r = dict(d)
    del r[key]
    return r


def dec2bin_sequence(x: torch.tensor, bits: int) -> torch.tensor:
    b = []
    for i in range(x.shape[0]):
        b.append(dec2bin(x[i].unsqueeze(0), BINARY_NUM))
    return torch.cat(b).flatten()

def dec2bin(x: torch.tensor, bits: int) -> torch.tensor:
    # calculate the sign bit
    sign = torch.tensor([[0 if x.signbit() == False else 1]])
    if x.signbit() == True:
        x = x.abs()
    # and the binary version of the num
    mask = 2 ** torch.arange(bits - 2, -1, -1).to(x.device, x.dtype)
    encoded = x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return torch.cat((sign, encoded), dim = 1)

# commented out as its outdated and we don't use it (hopefully ever). need to support the first "sign" bit if we want to get it working
# def bin2dec(b: torch.tensor, bits: int) -> torch.tensor:
#     mask = 2 ** torch.arange(bits - 2, -1, -1).to(b.device, b.dtype)
#     return torch.sum(mask * b, -1)

BINARY_NUM = 20