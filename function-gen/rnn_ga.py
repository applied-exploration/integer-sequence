

from models.ga.ga_plain import GA_Plain
from models.rnn.rnn_plain import RNN_Plain
from lang import Lang
from typing import List, Tuple
from learning_types import LearningAlgorithm
import sys
import os
import math
import random
from utils import eq_encoder, eq_decoder, is_eq_valid
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from models.rnn.combined_networks import Loss

class RNN_GA_Unified(LearningAlgorithm):

    def __init__(self, symbols: List[str], output_sequence_length: int, encoded_seq_length: int, mutation_rate: int, num_epochs_rnn: int, population_size: int, input_size: int, output_size: int, hidden_size: int = 256, learning_rate: float = 0.01, loss:Loss= Loss.NLL, num_epochs_ga: int = 20, seed:int = 1):
        random.seed(seed)

        self.symbols = symbols
        self.output_sequence_length = output_sequence_length
        self.encoded_seq_length = encoded_seq_length
        self.mutation_rate = mutation_rate
        self.num_epochs_rnn = num_epochs_rnn
        self.num_epochs_ga = num_epochs_ga
        self.population_size = population_size

        self.rnn = RNN_Plain(symbols=self.symbols, output_sequence_length=output_sequence_length, encoded_seq_length=encoded_seq_length, num_epochs=num_epochs_rnn,
                             input_size=input_size, hidden_size=hidden_size, output_size=output_size, loss = loss)

    def train(self, input_lang: Lang, output_lang: Lang, data: List[Tuple[List[int], str]]) -> None:
        self.rnn.train(input_lang, output_lang, data)

    def infer(self, input_lang: Lang, output_lang: Lang, data: List[List[int]]) -> List[str]:

        self.encoded_best_guesses = self.rnn.infer(input_lang, output_lang, data)
        print(self.encoded_best_guesses)
        self.decoded_best_guesses = [eq_encoder(output_eq) for output_eq in self.encoded_best_guesses]
        self.init_population = self.create_varied_population(self.decoded_best_guesses)

        self.ga = GA_Plain(symbols=self.symbols, output_sequence_length=self.output_sequence_length,
                           encoded_seq_length=self.encoded_seq_length, mutation_rate=self.mutation_rate, num_epochs=self.num_epochs_ga, population_size=self.population_size, init_population=self.init_population)

        pred = self.ga.infer(input_lang, output_lang, data)

        return pred

    def create_varied_population(self, decoded_best_guesses: List[List[int]]) -> List[List[List[int]]]:
        multiple_population: List[List[List[int]]] = []

        for best_guess in decoded_best_guesses:
            one_population: List[List[int]] = []

            for _ in range(self.population_size):
                mutant_best_guess = best_guess.copy()

                counter = 0
                valid = False
                while valid == False:
                    for i, gene in enumerate(mutant_best_guess):
                        if random.random() * 100 < self.mutation_rate:
                            mutant_best_guess[i] = math.floor(random.random() * len(list(self.symbols)))
                    
                    valid = is_eq_valid(eq_decoder(mutant_best_guess))

                    counter += 1
                    if counter > 25: 
                        mutant_best_guess = best_guess.copy()
                        break

                # valid = False
                # while valid == False:
                #     for i, gene in enumerate(mutant_best_guess):
                #         if random.random() * 100 < self.mutation_rate:
                #             mutant_best_guess[i] = math.floor(
                #                 random.random() * len(list(self.symbols)))

                #     valid = is_eq_valid(eq_decoder(mutant_best_guess))
            
            
                # for i, gene in enumerate(mutant_best_guess):
                #         if random.random() * 100 < self.mutation_rate:
                #             mutant_best_guess[i] = math.floor(
                #                 random.random() * len(list(self.symbols)))

                one_population.append(mutant_best_guess)

            multiple_population.append(one_population)

        return multiple_population
