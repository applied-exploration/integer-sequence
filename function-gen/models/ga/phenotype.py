import math
from .genotype import Genotype
from typing import List, Dict
from utils import eq_decoder, eq_to_seq

class Phenotype:

    def __init__(self, param, genotype: Genotype):
        self.param = param
        self.decoded_representation = eq_decoder(genotype.m_genes)

    
    def display(self):
        print(self.decoded_representation)


    def evaluate(self):
        target_sequence = self.param["target_sequence"]
        output = eq_to_seq(self.decoded_representation, len(self.param["target_sequence"])) 

        fitness: float = 0.0
        
        for i, value in enumerate(target_sequence):
            fitness -= min(1000, (value - output[i]))

        fitness /= len(target_sequence)

        #fitness += 0.0 # eg.: closeness to actual target value
        #fitness -= 0.0 # eg.: number of symbols

        return fitness

