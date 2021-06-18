from .phenotype import Phenotype
from .genotype import Genotype, crossover
from typing import List, Dict

class Individual:

    def __init__(self, param):
        
        self.param = param

        self.fitness : float = 0.0
        self.genotype: Genotype = Genotype(param)
        self.phenotype: Phenotype = Phenotype(param, self.genotype)

        self.evaluate()

    def display(self):
        self.phenotype.display()

    def evaluate(self):
        self.fitness = self.phenotype.evaluate()


def breed(param, a: Individual, b: Individual) -> Individual:
    c: Individual = Individual(param)

    c.genotype = crossover(param, a.genotype, b.genotype)
    c.genotype.mutate_()
    c.phenotype = Phenotype(param, c.genotype)

    return c

