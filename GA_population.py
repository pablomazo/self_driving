import numpy as np
from NeuralNetworks import FF1H
import torch

class GA_population():
    def __init__(self, pop_size, nparents, npermanent, nH):
        '''
        - gen: Generation.
        - pop_size: Number of individuals in population.
        - nH: Hidden layer.
        - nparents: Number of parents used to get the new generation.
        - npermanent: Number of individuals that will continue in the next generation.

        -w1, b1, w2, b2: Weights and bias as a vector of each individual
        '''

        self.gen = 1
        self.pop_size = pop_size
        self.nparents = nparents
        self.npermanent = npermanent

        self.nH = nH

        self.mating_pool = []
        self.population = []

        # Generate population.
        for individual in range(self.pop_size):
            net = FF1H(self.nH)
            net = net.eval()
            self.population.append(net)

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
                    tmp = fitness[i]
                    fitness[i] = fitness[j]
                    fitness[j] = tmp

        self.mating_pool = pop_order[:self.nparents]

    def get_new_generation(self, fitness):
        '''
        Function to generate the new generation.
        '''
        new_generation = []

        # Sort individuals by their fitness value:
        self.get_mating_pool(fitness)

        # Get the "npermantent" individuals with highest fitness value
        # to follow on the next generation.
        for i in range(self.npermanent):
            indv_id = self.mating_pool[i]
            new_generation.append(self.population[indv_id])

        # Get parents as the "nparents" individuals with highest 
        # fitness value
        parents = []
        for i in range(self.nparents):
            indv_id = self.mating_pool[i]
            parents.append(self.population[indv_id])

        # Generate the offspring:
        noffspring = self.pop_size - self.npermanent
        offspring = self.get_offspring(noffspring, parents)

        new_generation += offspring
        self.gen += 1

        self.population = new_generation

    def get_offspring(self, n, parents):
        '''
        Given the parents this method will execute crossovers and
        mutations of the net parameters for the next generation.
        '''

        offsprings = []

        nparents = len(parents)
        for offspring in range(n):
            # Random selection of parents:
            parent1 = np.random.randint(nparents)
            parent2 = np.random.randint(nparents)

            net = FF1H(self.nH)
            for param in net.state_dict().keys():
                if 'weight' in param:
                    weight1 = parents[parent1].state_dict()[param].numpy()
                    weight2 = parents[parent2].state_dict()[param].numpy()

                    new = self.crossover_weights(weight1, weight2)

                    new = self.mutate_weights(new)

                elif 'bias' in param:
                    bias1 = parents[parent1].state_dict()[param].numpy()
                    bias2 = parents[parent2].state_dict()[param].numpy()

                    new = self.crossover_bias(bias1, bias2)

                    new = self.mutate_bias(new)

                net.state_dict()[param] = torch.from_numpy(new)

            offsprings.append(net)

        return offsprings

    def crossover_weights(self, weights1, weights2):
        rows, columns = weights1.shape

        # Get principal parent:
        rand_ind = np.random.randint(2)
        if rand_ind == 0:
            new_weights = weights1.copy()
            secondary_parent = weights2
        else:
            new_weights = weights2.copy()
            secondary_parent = weights1

        # Permute a random number of rows:
        n_row_permutation = np.random.randint(rows)

        for irow in range(n_row_permutation):
            rand_row = np.random.randint(rows)
            new_weights[rand_row,:] = secondary_parent[rand_row,:]

        # Permute a random number of columns:
        n_col_permutation = np.random.randint(columns)

        for icol in range(n_col_permutation):
            rand_col = np.random.randint(columns)
            new_weights[:,rand_col] = secondary_parent[:,rand_col]

        return new_weights

    def crossover_bias(self, bias1, bias2):
        elems = bias1.shape[0]
        mean = int(elems / 2)

        new_bias = np.empty_like(bias1)
        new_bias[:mean] = bias1[:mean]
        new_bias[mean:] = bias2[mean:]

        return new_bias

    def mutate_weights(self, weights):
        '''
        Changes some of the parameters in "weigth" matrix to a random
        value. The number of mutations is randomly selected.
        '''

        rows, columns = weights.shape

        nmutations = np.random.randint(rows * columns)

        for mutation in range(nmutations):
            i = np.random.randint(rows)
            j = np.random.randint(columns)
            weights[i,j] = np.random.uniform(low=-1e0, high = 1e0)

        return weights

    def mutate_bias(self, bias):
        '''
        Changes some of the parameters in "bias" to a random
        value. The number of mutations is randomly selected.
        '''

        elems = bias.shape[0]

        nmutations = np.random.randint(elems)

        for mutation in range(nmutations):
            i = np.random.randint(elems)
            bias[i] = np.random.uniform(low=-1e0, high = 1e0)

        return bias
