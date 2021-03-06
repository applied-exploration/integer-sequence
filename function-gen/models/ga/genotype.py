import math
import random
from typing import List, Dict
from utils import generate_random_eq, is_eq_valid, eq_encoder, eq_decoder, generate_random_eq_valid


class Genotype:
    def __init__(self, param, init_genotype = None):
        self.param = param
        # self.m_genes: List[int] = [math.floor(random.random() * param['symbol_set_size']) for _ in range(param['encoded_seq_length'])]

        if init_genotype != None: 
            self.m_genes: List[int] = init_genotype
        else: 
            random_function = generate_random_eq_valid(param['encoded_seq_length'])
            encoded_function_to_genes = eq_encoder(random_function)
            self.m_genes: List[int] = encoded_function_to_genes


    def mutate_(self):
        # 5% mutation rate


        original_genes =  self.m_genes.copy()

        counter = 0
        valid = False
        while valid == False:
            for i, gene in enumerate(self.m_genes):
                if random.random() * 100 < self.param['mutation_rate']:
                    self.m_genes[i] = math.floor(random.random() * self.param['symbol_set_size'])
            
            valid = is_eq_valid(eq_decoder(self.m_genes))

            # print(counter)
            counter += 1
            if counter > 25: 
                self.m_genes = original_genes.copy()
                break



def crossover(param, a: Genotype, b: Genotype) -> Genotype:

    c: Genotype = Genotype(param)

    counter = 0

    valid = False
    while valid == False:
        for i, gene in enumerate(c.m_genes):
            if random.random() < 0.5:
                c.m_genes[i] = a.m_genes[i]
            else:
                c.m_genes[i] = b.m_genes[i]

        valid = is_eq_valid(eq_decoder(c.m_genes))

        counter += 1
        if counter > 25:
            if random.random() < 0.5:
                c.m_genes = a.m_genes.copy()
            else:
                c.m_genes = b.m_genes.copy()
            break

    return c
