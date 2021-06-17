from .individual import Individual, breed
import math
import random
from typing import List, Dict


class Population:

    def __init__(self, param, population_size: int = 100 ):

        self.param = param
        self.population_size = population_size
        self.m_pop: List[Individual] = [Individual(param)
                                        for _ in range(population_size)]

        # self.m_pop = make_valid(m_pop)

        self.m_pop.sort(key=lambda x: x.fitness)

    def evolve_(self):
        a: Individual = self.select()
        b: Individual = self.select()
        x: Individual = breed(self.param, a, b )

        # if is_valid(x): x = breed(a,b)

        x.evaluate()

        self.m_pop[0] = x
        self.m_pop.sort(key=lambda x: x.fitness)

    def select(self) -> Individual:
        index: int = math.floor(
            (float(self.population_size) - 1e-6) * (1.0 - math.pow(random.random(), 2)))
        return self.m_pop[index]
