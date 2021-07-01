
import gym
from random import randrange
from gym import error, spaces, utils
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from lang import Lang


class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

    def __str__(self):
        return "Value: " + str(self.val) + ", Left: " + str(self.left) + ", Right: " + str(self.right)


def insert(root, key):
    if root is None:
        return Node(key)
    elif root.left is None:
        root.left = insert(root.left, key)
    elif root.right is None:
        root.right = insert(root.right, key)
    return root


def generate_possibilities(char: str):
    def list_possibilities(prev: str) -> str:
        if prev in ['1','2','3','4','5','6','7','8','9','0']:
            return ['+', '-', '*']
        elif prev in ['s', '+', '-', '*']:
            return ['1','2','3','4','5','6','7','8','9','0','t']
        elif prev in ['t']:
            return ['+', '-', '*']
        else:
            raise ValueError('Unexpected prev character')
    lhs = list_possibilities(char)
    rhs = list_possibilities(char)
    return lhs, rhs




class IntegerSequenceEnv(gym.Env):  
    metadata = {'render.modes': ['human']}

    int_sequence: List[int]
    output_length: int
    target_function: str

    input_lang: Lang
    output_lang: Lang

    syms = list('+*-0123456789t')
    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(syms) }
    ix_to_char = { i:ch for i,ch in enumerate(syms) }

    eq_state: List[int] = []

    def __init__(self, int_sequence: List[int], target_function: str, input_lang: Lang, output_lang: Lang):
        self.action_space = spaces.Discrete(len(self.syms))
        # if this doesn't work, hard-code it
        self.observation_space = spaces.Tuple([spaces.Discrete(len(self.syms))] * len(target_function))
        self.int_sequence = int_sequence
        self.output_length = len(target_function)
        self.target_function = target_function
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.state = [int_sequence, []]


    def step(self, action):
        self.state = self.state + self.ix_to_char[action]
        if len(self.state) == self.output_length:
            if 
            
            return (self.__get_state(), reward, done)

        else:
            reward = 0
            done = False3453
            return (self.__get_state(), reward, done)
 
    def __get_state(self):
        return [self.int_sequence, self.eq_state]

    def reset(self, int_sequence: List[int], target_function: str, input_lang: Lang, output_lang: Lang):
        self.observation_space = spaces.Tuple([spaces.Discrete(len(self.syms))] * len(target_function))
        self.int_sequence = int_sequence
        self.output_length = len(target_function)
        self.target_function = target_function
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.state = [int_sequence, []]

    def render(self, mode='human', close=False):
        return self.__get_grid()
