import math
import random
from typing import List, Dict

class Genotype:
    def __init__(self, param):
        self.param = param
        self.m_genes: List[int] = [math.floor(random.random() * param['symbol_set_size']) for _ in range(param['encoded_seq_length'])]

    def mutate_(self):
        # 5% mutation rate

        for gene in self.m_genes:
            if random.random() * 100 < self.param['mutation_rate']:
                gene = math.random * self.param['symbol_set_size']


def crossover(param, a: Genotype, b: Genotype ) -> Genotype:
    c: Genotype = Genotype(param)

    for i, gene in enumerate(c.m_genes):
        if random.random() < 0.5:
            gene = a.m_genes[i]
        else:
            gene = b.m_genes[i]
        
    return c



