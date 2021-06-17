import math
from .genotype import Genotype
from typing import List, Dict

class Phenotype:

    def __init__(self, param, genotype: Genotype):
        self.param = param
        # here we should define the mapping of encoding to function
        self.decoded_representation = self.get_decoded(genotype)


    def get_decoded(self, genotype: Genotype):
        decoded_function: string = ''

        for gene in genotype.m_genes:
            decoded_function += self.param['symbols'][gene]

        return decoded_function
    
    def display(self):
        print(self.decoded_representation)

    def evaluate(self):
        fitness: float = 0.0

        fitness += 0.0 # eg.: closeness to actual target value
        fitness -= 0.0 # eg.: number of symbols

        return fitness