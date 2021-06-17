import math

class Genotype:
    def __init__(self, param):
        self.symbol_set_size = symbol_set_size
        self.m_genes: List[int] = [math.floor(math.random() * param.symbol_set_size) for _ in range(param.encoded_seq_length)]

    def mutate_(self):
        # 5% mutation rate

        for gene in self.m_genes:
            if math.random() * 100 < 5:
                gene = math.random * self.symbol_set_size


def crossover(a: Genotype, b: Genotype) -> Genotype:
    c: Genotype = Genotype()

    for i, gene in enumerate(c.m_genes):
        if math.random() < 0.5:
            gene = a.m_genes[i]
        else:
            gene = b.m_genes[i]
        
    return c



