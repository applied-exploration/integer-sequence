from sympy import *
from random import choice
from typing import List

def generate_random_eq(length: int) -> str:
    def generate_next(prev: str) -> str:
        if prev in ['1','2','3','4','5','6','7','8','9','0']:
            return choice(['+', '-', '*'])
        elif prev in ['s', '+', '-', '*']:
            return choice(['1','2','3','4','5','6','7','8','9','0', 't', 't', 't', 't', 'x', 'x', 'x', 'x', 'y', 'y', 'y', 'y'])
        elif prev in ['t', 'x', 'y']:
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
        results = [parse_expr(eq, local_dict = {'t': num}, evaluate=True) for num in test_set]
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
            prev_2 = int_seq[i-2] if i > 2 else 0
            prev_1 = int_seq[i-1] if i > 1 else 1
            int_seq.append(int(parse_expr(eq, local_dict = {'t': i+1, 'x': prev_1, 'y': prev_2 })))
        except:
            pass
    if len(int_seq) != length: return [0] * length
    return int_seq

syms = list('+*-0123456789txy')
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