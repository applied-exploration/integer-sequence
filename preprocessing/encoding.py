import torch
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

all_symbols = '0123456789,-'
n_symbols = len(all_symbols)

input_length = 207


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_symbols.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_symbols)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
# def lineToTensor(line):
#     tensor = torch.zeros(len(line), 1, n_symbols)
#     for li, letter in enumerate(line):
#         tensor[li][0][letterToIndex(letter)] = 1
#     return tensor

def seqToTensor(line):
    tensor = torch.zeros(input_length, n_symbols)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('-'))

print(seqToTensor('54,-2'))
print(seqToTensor('542').size())