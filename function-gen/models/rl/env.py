
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
import random

MAX_PENALTY_MAGNITUDE = 999.0

def index_of_first(lst, pred):
    for i,v in enumerate(lst):
        if pred(v):
            return i
    return None

TreeState = Tuple[List[int], List[int]]
Evaluate = Callable[[str, List[int]], float]

### Node class - for tree structure, not used just yet.
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

def list_possibilities(prev: str) -> List[str]:
    if prev in ['1','2','3','4','5','6','7','8','9','0']:
        return ['+', '-', '*']
    elif prev in ['', '+', '-', '*']:
        return ['1','2','3','4','5','6','7','8','9','0','t']
    elif prev in ['t']:
        return ['+', '-', '*']
    else:
        raise ValueError('Unexpected prev character', prev)

def encode_with_lang(lang: Lang, input):
    return [lang.word2index[word] for word in list(input)]

def decode_with_lang(lang: Lang, input):
    return [lang.index2word[word] for word in list(input)]

def create_initial_state(input_lang: Lang, data: List[List[int]], output_length: int) -> TreeState:
    int_seq = random.choice(data)[0]
    seq = encode_with_lang(input_lang, [str(i) for i in int_seq])
    return ([-1] * output_length, seq)


def get_current_position(state: TreeState) -> int:
    index = index_of_first(state[0], lambda x: x == -1)
    if index is None: return 0
    return index

def is_action_valid(state: TreeState, action: int, output_lang: Lang) -> bool:
    pos = get_current_position(state)
    if pos == 0: return True
    last_char = decode_with_lang(output_lang, [state[0][pos-1]])[0]
    possibilities = list_possibilities(last_char)
    next_char = decode_with_lang(output_lang, [action])[0]
    if next_char in possibilities:
        return True
    else:
        return False
    
    

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
    data: List[List[int]]   
    evaluate: Evaluate

    input_lang: Lang
    output_lang: Lang

    state: TreeState
    syms = list('+*-0123456789t')

    def __init__(self, env_config):
        self.output_length = env_config["output_length"]
        self.input_lang = env_config["input_lang"]
        self.output_lang = env_config["output_lang"]
        self.evaluate = env_config["evaluate"]
        self.data = env_config["data"]
        self.state = create_initial_state(self.input_lang, self.data, self.output_length)

        self.action_space = spaces.Discrete(len(self.syms))
        self.observation_space = spaces.Tuple((spaces.Box(low=-1, high=self.output_lang.n_words, shape=(self.output_length,), dtype= int), spaces.Box(low=0, high=self.input_lang.n_words, shape=(len(self.state[1]),), dtype= int)))



    def step(self, action):
        # offset action by 2, because we don't need SOS and EOS tokens here
        action = action + 2
        if not is_action_valid(self.state, action, self.output_lang):
            return (insert_action_in_state((self.state[0].copy(), self.state[1]), action), -MAX_PENALTY_MAGNITUDE, True, {})

        self.state = insert_action_in_state((self.state[0].copy(), self.state[1]), action)
        if is_state_complete(self.state):
            candidate_eq = ''.join(decode_with_lang(self.output_lang, self.state[0]))
            score = self.evaluate(candidate_eq, self.state[1])
            return (self.state, score, True, {})
        
        return (self.state, 0, False, {})
 

    def reset(self):
        self.state = create_initial_state(self.input_lang, self.data, self.output_length)
        return self.state

    def render(self, mode='human', close=False):
        return self.state
