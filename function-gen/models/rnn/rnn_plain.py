import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Tuple
from learning_types import LearningAlgorithm


import math
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .encoder_decoder_gru import EncoderRNN, DecoderRNN
from .combined_networks import train, infer, Loss
from .rnn_utils import tensorFromSentence, calc_magnitude
from utils import timeSince
from lang import Lang

import wandb


# from line_profiler import LineProfiler

# lp = LineProfiler()

EOS_token = 0
SOS_token = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN_Plain(LearningAlgorithm):

    def __init__(self, symbols: List[str], output_sequence_length: int, encoded_seq_length: int,  num_epochs: int, input_size:int, output_size:int, hidden_size: int = 256, embedding_size:int = 64, batch_size:int = 2, learning_rate: float = 0.01, num_gru_layers: int = 1, dropout_prob: float = 0.0, loss:Loss = Loss.NLL, bidirectional:bool=False, wandb_activate:bool=True, binary_encoding:bool = False, seed:int = 1):
        
        random.seed(seed)
        self.wandb_activate = wandb_activate

        self.symbols = symbols
        self.output_sequence_length = output_sequence_length
        self.encoded_seq_length = encoded_seq_length
        self.num_epochs = num_epochs

        self.learning_rate = learning_rate
        self.loss = loss

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.embedding_size, self.batch_size, num_gru_layers=num_gru_layers, dropout=dropout_prob, seed=seed, bidirectional=bidirectional, binary_encoding= binary_encoding).to(device)
        self.decoder = DecoderRNN(self.hidden_size, self.output_size, self.embedding_size, self.batch_size, num_gru_layers=num_gru_layers, dropout=dropout_prob, seed=seed, bidirectional_encoder=bidirectional).to(device)

        
        if self.wandb_activate: 
            wandb.watch(self.encoder, log_freq=100)
            wandb.watch(self.decoder, log_freq=100)
        
        print("Num_batch: ", self.batch_size)
        print(self.encoder)
        print(self.decoder)

    # WITH DATALOADER =>   
    # def convert_data(self, data:List[Tuple[List[int], str]]) -> List[Tuple[str, str]]:
    #     converted_data = []

    #     for pair in data:
    #         stringified_sequence = ''.join(str(x)+',' for x in pair[0])
    #         new_tuple = (stringified_sequence[:-1], pair[1])
    #         converted_data.append(new_tuple)

    #     return converted_data

    # def dataset_to_tensor(self, data: List[str], lang: Lang) -> List[torch.tensor]:
    #     return [tensorFromSentence(lang, data[i])
    #                     for i in range(len(data))]

    def seperate_data(self, data:List[Tuple[List[int], str]]) -> Tuple[List[str], List[str]]:
        ''' 
        Description: 
            Function that seperates input and target into seperate lists
        ---
        Input: 
        List of Tuples, that contain 
            - a list of integers (sequence we want to predict, eg.: [2, 4, 6, 8, 10, 12, 14, 16] - input) 
            - and a string (function we want to predict eg.: '5x3+t-2+8 - target)
        Output: 
            - list of inputs (string)
            - list of targets (string)
        '''

        input_data = []
        target_data = []

        for pair in data:
            stringified_sequence = ''.join(str(x)+',' for x in pair[0])
            input_data.append(stringified_sequence[:-1])
            target_data.append(pair[1])

        return input_data, target_data

    def create_minibatch(self, data: List[str], lang: Lang, indices:List[int]) -> torch.tensor:
        encoded_dataset = [tensorFromSentence(lang, data[index]) for index in indices]
        
        ## flatten it to a batch tensor, one_column = one batch of sequence, one_row = time step in sequences
        return torch.cat(encoded_dataset, dim=1) 



    def train(self, input_lang: Lang, output_lang: Lang, data: List[Tuple[List[int], str]]) -> None:
        print_every = max(1, math.floor(self.num_epochs/10))

        ''' For diagnosis'''
        start = time.time()
        print_loss_total = 0  # Reset every print_every


        ''' Defining Optimization parameters'''
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

        ''' 
        NLLLos requires 
            - an input tensor of negative logprobabilities 
              shaped [batch_size, category size (how many symbols we have)] 
              eg.: [[ -2.1, -1.2 ]
                    [ -1.5, -0.5 ]
                    [ -0.7. -0.3 ]]  with num_batch 3 and category size 2 (if eg.: we had only symbols '*+')
            - and a target tensor 
              shaped [num_batches]. 
              eg.: [2, 5, 15] with num_batch 3. Each number represents a category
        '''
        criterion = nn.NLLLoss()  

        ''' Prepare Data '''
        input_data, target_data = self.seperate_data(data)

        # --- with DataLoader --- # 
        # train_dataloader = DataLoader((self.dataset_to_tensor(input_data, input_lang),self.dataset_to_tensor(target_data, output_lang)),
        #      batch_size=self.batch_size, shuffle=True)
        # input_tensor, target_tensor = next(iter(train_dataloader))

        

        ''' Feed forward of network & calculating loss'''
        for i in range(1,  self.num_epochs + 1):
            
            ''' Create a minibatch tensor [sequence_len, batch_size]'''
            # --- with own minibatching --- #
            randomized_indices = [random.randrange(0, len(data)) for _ in range(0, self.batch_size)]
            input_tensor_minibatch = self.create_minibatch(input_data, input_lang, randomized_indices)
            target_tensor_minibatch = self.create_minibatch(target_data, output_lang, randomized_indices)
 
            # --- with DataLoader --- #
            # input_tensor, target_tensor = next(iter(train_dataloader))
                  

            loss = train(
                input_tensor_minibatch, target_tensor_minibatch, 
                self.encoder, self.decoder, 
                encoder_optimizer, decoder_optimizer, 
                criterion, 
                input_lang, output_lang, 
                loss_type = self.loss )

            print_loss_total += loss

            ''' Print diagnostic '''
            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                if self.wandb_activate: wandb.log({'loss': print_loss_avg, 'epoch': i})
                print('%s (%d %d%%) %.4f' % (timeSince(start, i / self.num_epochs),
                                            i, i / self.num_epochs * 100, print_loss_avg))



    def infer(self, input_lang: Lang, output_lang: Lang, data: List[List[int]]) -> List[str]:
        ''' Prepare data '''
        stringified_inputs = [''.join(str(x)+',' for x in sequence) for sequence in data]
        
        input_tensor_batch = self.create_minibatch(stringified_inputs, input_lang, list(range(0, len(data))))
        output_list = infer(input_tensor_batch, self.encoder, self.decoder, output_lang )

        return output_list
        
    def save(self, name: str):
        folder = ""
        torch.save(self.encoder.state_dict(), name + "-encoder.pt")
        torch.save(self.decoder.state_dict(), name + "-decoder.pt")

