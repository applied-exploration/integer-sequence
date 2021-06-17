from individual import Individual, breed
import math


class Population:

    def __init__(self, population_size):
        self.m_pop: List[Individual] = [Individual()
                                        for _ in range(population_size)]

        # self.m_pop = make_valid(m_pop)
        self.m_pop.sort(key=lambda x: x.m_fitness)

    def evolve_(self):
        a: Individual = self.select()
        b: Individual = self.select()
        x: Individual = breed(a, b)

        # if is_valid(x): x = breed(a,b)

        x.evaluate()

        self.m_pop[0] = x
        self.m_pop.sort(key=lambda x: x.m_fitness)

    def select(self) -> Individual:
        index: int = math.floor(
            (100.0 - 1e-6) * (1.0 - math.pow(math.random(), 2)))
        return self.m_pop[index]
