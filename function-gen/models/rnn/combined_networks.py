import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from typing import List, Tuple
from learning_types import LearningAlgorithm
from lang import Lang
from utils import accuracy_score, mae_score
import wandb
from enum import Enum
import random
from .rnn_utils import calc_magnitude

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5
EOS_token = 0
SOS_token = 1
MAX_LENGTH = 10

''' Trains the LearningAlgorithm and reports every 1000 epoch'''
def train_report(algo: LearningAlgorithm, input_lang: Lang, output_lang: Lang, training_data: List[Tuple[List[int], str]], test_data_X: List[List[int]], test_data_y: List[str], num_epochs: int) -> None:
    num_batches = max(1, int(num_epochs / 1000))
    algo.num_epochs = 1000

    sampled_test_X = test_data_X[:1000]
    sampled_test_y = test_data_y[:1000]

    for _ in range(0, num_batches):
        algo.train(input_lang, output_lang, training_data)
        pred_test = algo.infer(input_lang, output_lang, sampled_test_X)
        accuracy_test = accuracy_score(pred_test, sampled_test_y)
        mae_test = mae_score(pred_test, sampled_test_y)
        print("Accuracy score on test set: ", accuracy_test)
        print("Mean Absolute Error  on test set: ", mae_test)
        wandb.log({
            'accuracy': accuracy_test,
            'mae': mae_test
        })


class Loss(Enum):
    NLL = 1
    NLL_Plus_MAE = 2
    NLL_Multiply_MAE = 3


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_lang, output_lang, with_attention = False, loss_type: Loss = Loss.NLL, max_length=MAX_LENGTH, seed = 1):


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    batch_size_inferred = input_tensor.shape[1]

    ''' ENCODER '''
    encoder_hidden = encoder.initHidden()
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    
    ''' DECODER '''
    decoder_input = torch.tensor([[SOS_token for _ in range(batch_size_inferred)]], device=device)

    # decoder_hidden = decoder.initHidden(encoder_hidden)
    encoder_out = encoder_hidden

    decoder_hidden = encoder_hidden.unsqueeze(0)


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    decoder_outputs = []

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if with_attention == True:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            decoder_outputs.append(decoder_output)
            decoder_squeezed = decoder_output.squeeze(0)

            loss += criterion(decoder_squeezed, target_tensor[di])
            decoder_input = target_tensor[di].unsqueeze(0)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if with_attention == True:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            ''' Here we are selecting the top prediction for each batch '''
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(1, -1)  # detach from history as input  

            
            decoder_output_squeezed = decoder_output.squeeze(0)
            loss += criterion(decoder_output_squeezed, target_tensor[di])

    if loss_type != Loss.NLL:
        sliced_decoder_outputs = [[dec_output[:,i] for dec_output in decoder_outputs] for i in range(0, batch_size_inferred)]
        magnitudes = []
        if loss_type == Loss.NLL_Plus_MAE:
            magnitudes = [calc_magnitude(sliced_decoder_outputs[i], target_tensor[:,i].unsqueeze(1), output_lang, 19) for i in range(0, batch_size_inferred)]
            loss = loss + (sum(magnitudes) / len(magnitudes))

        elif loss_type == Loss.NLL_Multiply_MAE:
            magnitudes = [calc_magnitude(sliced_decoder_outputs[i], target_tensor[:,i].unsqueeze(1), output_lang, 9) for i in range(0, batch_size_inferred)]
            loss = loss * (sum(magnitudes) / len(magnitudes))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def infer(input_tensor, encoder, decoder, output_lang, with_attention = False):
    max_length=  10
    output_list = []

    input_length = input_tensor.size(0)
    batch_size_inferred = input_tensor.shape[1]

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    with torch.no_grad():

        ''' ENCODER '''
        encoder_hidden = encoder.initHidden(batch_size = batch_size_inferred)
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)


        ''' DECODER '''
        decoder_input = torch.tensor([[SOS_token for _ in range(batch_size_inferred)]], device=device)
        decoder_hidden = encoder_hidden

        for di in range(max_length):
            if with_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  
                    
            topv, topi = decoder_output.data.topk(1)

            decoder_input = topi.squeeze().detach().view(1,-1)
            decoded_words.append(decoder_input)
            
        
        ''' PROCESS OUTPUT '''
        concatenated_output_sequences = torch.cat(decoded_words, dim=0).transpose(0,1)

        for output in concatenated_output_sequences.cpu().numpy():
            word = []
            for character in output:
                output_decoded = output_lang.index2word[character]
                word.append(output_decoded)
            
            stringified_output = ''.join(word[:-1])
            output_list.append(stringified_output)


    return output_list #decoded_words, output_sequence, decoder_attentions[:di + 1]
