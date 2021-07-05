
import gym
# from random import randrange
from gym import error, spaces, utils
from typing import Callable, List, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from lang import Lang
from utils import normalize_0_1, eq_to_seq, is_eq_valid


def index_of_first(lst, pred):
    for i,v in enumerate(lst):
        if pred(v):
            return i
    return None

TreeState = Tuple[List[int], List[int]]

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
    def list_possibilities(prev: str) -> List[str]:
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

def encode_with_lang(lang: Lang, input):
    return [lang.word2index[word] for word in list(input)]

def decode_with_lang(lang: Lang, input):
    return [lang.index2word[word] for word in list(input)]

def create_initial_state(input_lang: Lang, int_seq: List[int], output_length: int) -> TreeState:
    seq = encode_with_lang(input_lang, [str(i) for i in int_seq])
    return ([-1] * output_length, seq)


def get_current_position(state: TreeState) -> int:
    index = index_of_first(state[0], lambda x: x == -1)
    if index is None: return 0
    return index

def insert_action_in_state(state: TreeState, action: int) -> TreeState:
    pos = get_current_position(state)
    eq_state = state[0].copy()
    eq_state[pos] = action
    return (eq_state, state[1])

def is_state_complete(state: TreeState) -> bool:
    if -1 in set(state[0]):
        return False
    else:
        return True



class IntegerSequenceEnv(gym.Env):  
    output_length: int
    target_function: str
    evaluate_tree: Callable[[str, str], float]

    input_lang: Lang
    output_lang: Lang

    state: TreeState
    syms = list('+*-0123456789t')

    def __init__(self, int_sequence: List[int], target_function: str, input_lang: Lang, output_lang: Lang, evaluate_tree: Callable[[str, str], float]):
        self.action_space = spaces.Discrete(len(self.syms))
        # self.observation_space = spaces.Tuple([spaces.Discrete(len(self.syms))] * len(target_function))
        self.output_length = len(target_function)
        self.target_function = target_function
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.evaluate_tree = evaluate_tree
        self.state = create_initial_state(self.input_lang, int_sequence, self.output_length)


    def step(self, action):
        # offset action by 2, because we don't need SOS and EOS tokens here
        self.state = insert_action_in_state((self.state[0].copy(), self.state[1]), action + 2)
        if is_state_complete(self.state):
            candidate_eq = ''.join(decode_with_lang(self.output_lang, self.state[0]))
            score = self.evaluate_tree(candidate_eq, self.target_function)
            return (self.state, score, True)
        
        return (self.state, 0, False)
 

    def reset(self, int_sequence: List[int], target_function: str, input_lang: Lang, output_lang: Lang):
        # self.observation_space = spaces.Tuple([spaces.Discrete(len(self.syms))] * len(target_function))
        self.int_sequence = int_sequence
        self.output_length = len(target_function)
        self.target_function = target_function
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.state = create_initial_state(self.input_lang, int_sequence, self.output_length)

    def render(self, mode='human', close=False):
        return self.state
