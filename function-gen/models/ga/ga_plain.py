
from .population import Population
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from learning_types import LearningAlgorithm
from typing import List, Tuple

class GA_Plain(LearningAlgorithm):

    def __init__(self, symbols: List[str], output_sequence_length: int, encoded_seq_length: int, mutation_rate: int, num_epochs: int, population_size: int):
        self.symbols = symbols
        self.output_sequence_length = output_sequence_length
        self.encoded_seq_length = encoded_seq_length
        self.mutation_rate = mutation_rate
        self.num_epochs = num_epochs
        self.population_size = population_size

    def train(self, data: List[Tuple[List[int], str]]) -> None:
        pass

    def infer(self, data: List[List[int]]) -> List[str]:
        output: List[str] = []
        for i, seq in enumerate(data):
            param =  {
                "symbols": self.symbols,
                "symbol_set_size": len(list(self.symbols)),
                "encoded_seq_length" : self.encoded_seq_length,
                "mutation_rate": self.mutation_rate,
                "output_sequence_length": self.output_sequence_length,
                "target_sequence": seq[0],
            }
            new_pop = Population(param, self.population_size)
            for _ in range(self.num_epochs):
                new_pop.evolve_()
                # print(new_pop.m_pop[-1].fitness)
                # print(new_pop.m_pop[0].fitness)
                if new_pop.m_pop[-1].fitness == 0.0: break
            # print('output_equation ', new_pop.m_pop[-1].phenotype.decoded_representation)
            output.append(new_pop.m_pop[-1].phenotype.decoded_representation)
        return output

