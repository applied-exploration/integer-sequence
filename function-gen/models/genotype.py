import math
import random
from typing import List, Dict
from utils import generate_random_eq, is_eq_valid, eq_encoder, eq_decoder, generate_random_eq_valid


class Genotype:
    def __init__(self, param):
        self.param = param
        # self.m_genes: List[int] = [math.floor(random.random() * param['symbol_set_size']) for _ in range(param['encoded_seq_length'])]

        random_function = generate_random_eq_valid(param['encoded_seq_length'])
        encoded_function_to_genes = eq_encoder(random_function)
        self.m_genes: List[int] = generate_random_eq

    def mutate_(self):
        # 5% mutation rate

        valid = False
        while valid == False:
            for gene in self.m_genes:
                if random.random() * 100 < self.param['mutation_rate']:
                    gene = math.random * self.param['symbol_set_size']
            
            valid = is_eq_valid(eq_decoder(self.m_genes, [1, 2, 4, 5, 10]))



def crossover(param, a: Genotype, b: Genotype) -> Genotype:

    c: Genotype = Genotype(param)

    valid = False
    while valid == False:
        for i, gene in enumerate(c.m_genes):
            if random.random() < 0.5:
                gene = a.m_genes[i]
            else:
                gene = b.m_genes[i]

        valid = is_eq_valid(eq_decoder(gene.m_genes, [1, 2, 4, 5, 10]))

    return c
