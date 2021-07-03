import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Tuple
from learning_types import LearningAlgorithm


import math
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

from .encoder_decoder_gru import EncoderRNN, DecoderRNN
from .combined_networks import train
from .rnn_utils import tensorsFromPair, tensorFromSentence, calc_magnitude
from utils import showPlot, timeSince, asMinutes
# from lang import load_data
from lang import Lang



# MAX_LENGTH = 10
EOS_token = 0
SOS_token = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN_Plain(LearningAlgorithm):

    def __init__(self, symbols: List[str], output_sequence_length: int, encoded_seq_length: int,  num_epochs: int, input_size:int, output_size:int, hidden_size: int = 256, embedding_size:int = 64, learning_rate: float = 0.01, calc_magnitude_on = False):
        self.symbols = symbols
        self.output_sequence_length = output_sequence_length
        self.encoded_seq_length = encoded_seq_length
        self.num_epochs = num_epochs

        self.learning_rate = learning_rate

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.embedding_size).to(device)
        self.decoder = DecoderRNN(self.hidden_size, self.output_size, self.embedding_size).to(device)

        if calc_magnitude_on:
            self.calc_magnitude = calc_magnitude
        else:
            self.calc_magnitude = None

    def convert_data(self, data:List[Tuple[List[int], str]]) -> List[Tuple[str, str]]:
        converted_data = []

        for pair in data:
            stringified_sequence = ''.join(str(x)+',' for x in pair[0])
            new_tuple = (stringified_sequence[:-1], pair[1])
            converted_data.append(new_tuple)

        return converted_data

    def randomize_seperate_data(self, data:List[Tuple[List[int], str]]) -> Tuple[List[str], List[str]]:
        input_data = []
        target_data = []

        shuffled_data = random.sample(data, len(data))


        for pair in shuffled_data:
            stringified_sequence = ''.join(str(x)+',' for x in pair[0])
            # new_tuple = (stringified_sequence[:-1], pair[1])
            # converted_data.append(new_tuple)

            input_data.append(stringified_sequence[:-1])
            target_data.append(pair[1])

        return input_data, target_data

    def create_minibatch(self, data, batch_size, num_epochs, lang):
        ''' Create a dataset that has num_epochs of minibatches which include randomly selected pairs'''
        n_dataset = batch_size * num_epochs
        
        # truncated_dataset = data[:n_dataset]
        truncated_dataset = [tensorFromSentence(lang, data)
                        for i in range(n_dataset)]

        print("truncated_dataset")
        print(truncated_dataset)


        minibatch_dataset = []

        for i in range(0, num_epochs):
            minibatch_dataset.append(truncated_dataset[i*batch_size:(i+1)*batch_size])

        for i, minibatch in enumerate(minibatch_dataset):
            concatenated_pair = torch.cat(*minibatch)
            print(concatenated_pair)
        # print(minibatch_dataset)
        # for i in range(0, n_dataset, batch_size):
        #     minibatch_dataset
        #     print(truncated_dataset[i])

        return minibatch_dataset

    def train(self, input_lang: Lang, output_lang: Lang, data: List[Tuple[List[int], str]]) -> None:
        print_every = max(1, math.floor(self.num_epochs/10))
        # plot_every=100

        ''' For diagnosis'''
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        ''' Prepare data'''
        ## OLD ##
        pairs = self.convert_data(data)
        training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
                        for i in range(self.num_epochs)]

        ## NEW ##
        input_data, target_data = self.randomize_seperate_data(data)
        print("original pairs")
        print(pairs[:3])
        print("input data")
        print(input_data[:3])
        print("target data")
        print(target_data[:3])
        minibatch_dataset_input = self.create_minibatch(input_data, 3, self.num_epochs, input_lang)
        minibatch_dataset_target = self.create_minibatch(target_data, 3, self.num_epochs, output_lang)


        ''' Defining Optimization parameters'''
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

        criterion = nn.NLLLoss()  # converted_loss

        # print("training pairs[:5]")
        # print(training_pairs[:5])

        ''' Feed forward of network & calculating loss'''
        for iter in range(1,  self.num_epochs + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            loss = train(input_tensor, target_tensor, self.encoder, self.decoder, encoder_optimizer, decoder_optimizer, criterion, input_lang, output_lang, calc_magnitude = self.calc_magnitude )
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / self.num_epochs),
                                            iter, iter / self.num_epochs * 100, print_loss_avg))


                # if iter % plot_every == 0:
                #     plot_loss_avg = plot_loss_total / plot_every
                #     plot_losses.append(plot_loss_avg)
                #     plot_loss_total = 0


    def infer(self, input_lang: Lang, output_lang: Lang, data: List[List[int]]) -> List[str]:
        max_length=  10
        output_list = []

        for input_sequence in data:
            sentence = ''.join(str(x)+',' for x in input_sequence)
            sentence = sentence[:-1]

            with torch.no_grad():
                input_tensor = tensorFromSentence(input_lang, sentence)
                input_length = input_tensor.size()[0]
                encoder_hidden = self.encoder.initHidden()

                encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)

                for ei in range(input_length):
                    encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                            encoder_hidden)
                    encoder_outputs[ei] += encoder_output[0, 0]

                decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

                decoder_hidden = encoder_hidden

                decoded_words = []
                decoder_attentions = torch.zeros(max_length, max_length)

                for di in range(max_length):
                    # decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    #     decoder_input, decoder_hidden, encoder_outputs)
                    # decoder_attentions[di] = decoder_attention.data

                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)  # this if or simply decoder
                        
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(output_lang.index2word[topi.item()])

                    decoder_input = topi.squeeze().detach()

                stringified_output = ''.join(decoded_words[:-1])
                output_list.append(stringified_output)
                # output_sequence = eq_to_seq(stringified_output, 9)

        return output_list #decoded_words, output_sequence, decoder_attentions[:di + 1]
        
