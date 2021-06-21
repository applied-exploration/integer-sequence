
from typing import List, Tuple
from learning_types import LearningAlgorithm
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import math
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

from models.rnn.encoder_decoder_gru import EncoderRNN, DecoderRNN
from models.rnn.combined_networks import train
from rnn_utils import tensorsFromPair
from utils import showPlot, timeSince, asMinutes
# from lang import load_data
from lang import Lang

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN_Plain(LearningAlgorithm):

    def __init__(self, symbols: List[str], output_sequence_length: int, encoded_seq_length: int,  num_epochs: int, input_size:int, output_size:int, hidden_size: int = 256, learning_rate: float = 0.01, calc_magnitude = None):
        self.symbols = symbols
        self.output_sequence_length = output_sequence_length
        self.encoded_seq_length = encoded_seq_length
        self.num_epochs = num_epochs

        self.learning_rate = learning_rate

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = EncoderRNN(self.input_size, self.hidden_size).to(device)
        self.decoder = DecoderRNN(self.hidden_size, self.output_size).to(device)

        self.calc_magnitude=calc_magnitude

    def convert_data(self, data:List[Tuple[List[int], str]]) -> List[Tuple[str, str]]:
        converted_data = []

        for pair in data:
            stringified_sequence = ''.join(str(x)+',' for x in pair[0])
            new_tuple = (stringified_sequence, pair[1])
            converted_data.append(new_tuple)

        return converted_data

    def train(self, data: List[Tuple[List[int], str]], input_lang: Lang, output_lang: Lang) -> None:
        print_every=1000
        # plot_every=100

        pairs = self.convert_data(data)

        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)
        training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
                        for i in range(self.num_epochs)]
        criterion = nn.NLLLoss()  # converted_loss

        for iter in range(1,  self.num_epochs + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, self.encoder,
                        self.decoder, encoder_optimizer, decoder_optimizer, criterion, input_lang, output_lang, self.calc_magnitude )
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


    def infer(self, data: List[List[int]]) -> List[str]:
        pass
        # output: List[str] = []
        # for i, seq in enumerate(data):
        #     param = {
        #         "symbols": self.symbols,
        #         "symbol_set_size": len(list(self.symbols)),
        #         "encoded_seq_length": self.encoded_seq_length,
        #         "mutation_rate": self.mutation_rate,
        #         "output_sequence_length": self.output_sequence_length,
        #         "target_sequence": seq[0],
        #     }
        #     new_pop = Population(param, self.population_size)
        #     for _ in range(self.num_epochs):
        #         new_pop.evolve_()
        #         # print(new_pop.m_pop[-1].fitness)
        #         # print(new_pop.m_pop[0].fitness)
        #         if new_pop.m_pop[-1].fitness == 0.0:
        #             break
        #     # print('output_equation ', new_pop.m_pop[-1].phenotype.decoded_representation)
        #     output.append(new_pop.m_pop[-1].phenotype.decoded_representation)
        # return output
