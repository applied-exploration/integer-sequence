from sympy import *
from random import choice, randrange
import numpy as np
import pandas as pd
from typing import List

def generate_random_eq(length: int) -> str:
    def generate_next(prev: str) -> str:
        if prev in ['1','2','3','4','5','6','7','8','9','0']:
            return choice(['+', '-', '*'])
        elif prev in ['s', '+', '-', '*']:
            return choice(['1','2','3','4','5','6','7','8','9','0', 't', 't', 't', 't'])
        elif prev == 't':
            return choice(['+', '-', '*'])
        else:
            raise ValueError('Unexpected prev character')

    result: List[str] = []
    for i in range(0, length):
        last_char = generate_next('s') if i == 0 else generate_next(result[i-1])
        result.append(last_char)
    return ''.join(result)

def is_eq_valid(eq: str, test_set: List[int]) -> bool:
    try:
        if "t" not in eq:
            return False
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

