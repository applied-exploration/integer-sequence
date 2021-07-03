import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os 

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils import normalize_0_1, eq_to_seq, is_eq_valid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EOS_token = 0
SOS_token = 1


def indexesFromSentence(lang, sentence):
    if ',' in sentence: 
        return [lang.word2index[word] for word in sentence.split(',') if word is not '']
    else:
        return [lang.word2index[word] for word in list(sentence)]



def tensorFromSentence(lang, sentence):
    print(sentence)
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)





def compare_sequences(output_sequence: np.ndarray, target_sequence: np.ndarray) -> float:
        magnitude: float = 0.0

        combined_seq = np.array([output_sequence, target_sequence]) 
        norm_comb_seq = normalize_0_1(combined_seq)

        norm_output_seq = norm_comb_seq[0]
        norm_target_seq = norm_comb_seq[1]
  
        
        for i, value in enumerate(norm_target_seq.tolist()):
            magnitude += abs(value - norm_output_seq[i])#**2

        # magnitude /= len(norm_target_seq)

        return torch.tensor(magnitude)



def calc_magnitude(decoder_outputs, target_outputs, input_lang, output_lang):
    max_penalty_magnitude = torch.tensor(9., dtype=torch.float64)
  
    decoded_output_symbols = []
    decoded_target_symbols = []
    detached_target_outputs = target_outputs.cpu().detach().numpy().squeeze()

    for decoder_output in decoder_outputs:
        topv, topi = decoder_output.data.topk(1)
        decoded_output = output_lang.index2word[topi.item()]
        decoded_output_symbols.append(decoded_output)
    
    for i, target_output in enumerate(detached_target_outputs):
        decoded_target = output_lang.index2word[target_output]
        decoded_target_symbols.append(decoded_target)
    
    stringified_output = ''.join(decoded_output_symbols)
    
    if is_eq_valid(stringified_output) == False:
        return max_penalty_magnitude

    output_sequence = eq_to_seq(stringified_output, 9)

    if np.count_nonzero(output_sequence) < 1:
        return max_penalty_magnitude
    else:
        stringified_target = ''.join(decoded_target_symbols[:-1])
        target_sequence = eq_to_seq(stringified_target, 9)

    return compare_sequences(np.array(output_sequence), np.array(target_sequence))

