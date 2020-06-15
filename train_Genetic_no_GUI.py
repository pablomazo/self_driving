from Controller import Controller
from GA_population import GA_population
from NeuralNetworks import save_model
import numpy as np
import sys
import torch


#Instantiate genetic algorithm
population = int(sys.argv[1])
n_father = int(sys.argv[2])
npermanent = int(sys.argv[3])
nH = int(sys.argv[4])
genetic = GA_population(population,n_father, npermanent,nH)

#Instanciate Controller
controller = Controller()
controller.load_circuit()
controller.register_genetic(genetic, GUI=False)

n_change = 20
best_fitness = 0
for generation in range(100000):
    try:
        print('Generation:', generation)
        #if generation % n_change == 0:
        #    controller.load_circuit()
        controller.reset()
        done = False

        while not done:

            crashed = []
            for i, player in enumerate(controller.players):
                crashed.append(player.crashed)
                end = player.count >= 100 or player.laps >= 10
                if end: player.crashed = True
                if not player.crashed:

                    # Get keys pressed by players.
                    controller.set_state(player)
                    key = player.handle_keys()
                    controller.exec_action(player, key)

            done = all(crashed)


        # Get number of blocks each car moved to act as fitness function:
        fitness = [controller.players[i].car.block + controller.players[i].laps * controller.circuit.nblocks for i in range(genetic.pop_size)]
        print('Fitness:', fitness)

        # Get id of best individual:
        best = np.argmax(fitness)

        if fitness[best] > best_fitness:
            save_model(controller.players[best].network, './saved_models/genetic_{}.pth'.format(generation))
            best_fitness = fitness[best]

        # Update parameters of each individual:
        genetic.get_new_generation(fitness)

    except KeyboardInterrupt:
        # Save best individual before exiting:
        # Get number of blocks each car moved to act as fitness function:
        fitness = [controller.players[i].car.block + controller.players[i].laps * controller.circuit.nblocks for i in range(genetic.pop_size)]
        print('Current fitness before exiting:')
        print(fitness)

        # Get id of best individual:
        best = np.argmax(fitness)

        # Save ckeckpoint of best individual.
        save_model(controller.players[best].network, './saved_models/checkpoint_genetic.pth')
        print('Best player before exit saved in checkpoint_genetic.pth with fitness:', fitness[best])

        sys.exit()
