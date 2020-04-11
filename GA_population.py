import numpy as np
from NeuralNetwork import NeuralNetwork

class GA_population():
    def __init__(self, pop_size, nparents, nI, nH, nO):
        '''
        - gen: Generation.
        - pop_size: Number of individuals in population.
        - nI, nH, nO: Size of input, hidden and ouptut layer.
        - nparents: Number of parents used to get the new generation.

        -w1, b1, w2, b2: Weights and bias as a vector of each individual
        '''

        self.gen = 1
        self.pop_size = pop_size
        self.nparents = nparents

        self.nI = nI
        self.nH = nH
        self.nO = nO

        self.mating_pool = []
        self.population = []

        # Generate population.
        for individual in range(self.pop_size):
            net = NeuralNetwork(nI, nH, nO)
            self.random_initialize(net)
            self.population.append(net)

    def random_initialize(self, net):
        '''
        Random initialization of the net weights and bias.
        '''
        net.w_input = self.generate_gene([self.nI, self.nH])
        net.b_input = self.generate_gene(self.nH)

        net.w_output = self.generate_gene([self.nH, self.nO])
        net.b_output = self.generate_gene(self.nO)

    def generate_gene(self, dimension, low=-1e0, high=1e0):
        '''
        Generates a random gene of dimension "dimension".
        '''
        return np.random.uniform(size=dimension, low=low, high=high)

    def get_mating_pool(self, fitness):
        '''
        Sorts the population by their fitness value and returns
        th and returns "nparents" elements with highest value of
        fitness function.
        '''

        pop_order = [i for i in range(self.pop_size)]

        for i in range(self.pop_size):
            for j in range(i+1,self.pop_size):
                if fitness[j] > fitness[i]:

                    # Save order from highest to lowest fitness value
                    tmp = pop_order[i]
                    pop_order[i] = pop_order[j]
                    pop_order[j] = tmp

                    # Sort array of fitness values.
                    tmp = fitness[i].copy()
                    fitness[i] = fitness[j].copy()
                    fitness[j] = tmp.copy()

        self.mating_pool = pop_order[:self.nparents]

