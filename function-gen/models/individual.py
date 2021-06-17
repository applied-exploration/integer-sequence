from phenotype import Phenotype
from genotype import Genotype, crossover

class Individual:

    def __init__(self, population_size):
        
        self.fitness : float = 0.0
        self.genotype: Genotype = Genotype()
        self.phenotype: Phenotype = Phenotype(self.genotype)

        self.evaluate()

    def display(self):
        self.phenotype.display()

    def evaluate(self):
        self.fitness = self.phenotype.evaluate()


def breed(a: Individual, b: Individual) -> Individual:
    c: Individual = Individual()

    c.m_genotype = crossover(a.m_genotype, b.m_genotype)
    c.m_genotype.mutate()
    c.m_phenotype = Phenotype(c.m_genotype)

    return c

