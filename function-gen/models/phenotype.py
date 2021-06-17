import math

class Phenotype:

    def __init__(self, genotype):
        # here we should define the mapping of encoding to function
        pass


    def display(self):
        pass

    def evaluate(self):
        fitness: float = 0.0

        fitness += 0.0 # eg.: closeness to actual target value
        fitness -= 0.0 # eg.: number of symbols

        return fitness