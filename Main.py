from GA_population import GA_population
import pygame
import numpy as np
from Controller import Controller
import sys

#Create Window
WHITE = (255,255,255)
BLUE = (0,0,255)
BLACK = (0,0,0)

screen_size = (0, 0)
pygame.init()
screen = pygame.display.set_mode(screen_size)
done = False

#Instantiate genetic algorithm
population = int(sys.argv[1])
n_father = int(sys.argv[2])
npermanent = int(sys.argv[3])
nI = int(sys.argv[4])
nH = int(sys.argv[5])
nO = int(sys.argv[6])
genetic = GA_population(population,n_father, npermanent,nI,nH,nO)

#Instanciate Controller
controller = Controller()
controller.register_genetic(genetic)
controller.initialize_genetic_players()

crashed = [False] * genetic.pop_size


clock = pygame.time.Clock()
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill(BLACK)

    #Repainting all circuit
    for block in range(controller.circuit.nblocks):
        x0, _, y0, _ = controller.circuit.get_block_coor(block)
        rect = pygame.Rect(x0, y0, controller.circuit.width, controller.circuit.height)

        if block == 0:
            pygame.draw.rect(screen, BLUE, rect)
        else:
            pygame.draw.rect(screen, WHITE, rect)

    for i in range(genetic.pop_size):
        crashed[i] = controller.players[i].crashed
        if controller.players[i].count >= 100:controller.players[i].crashed = True
        if not controller.players[i].crashed:
            #Repainting car
            controller.players[i].draw(screen)
             
            # Get keys pressed by players.
            controller.set_state(controller.players[i])
            key = controller.players[i].handle_keys()
            controller.exec_action(controller.players[i], key)
            controller.is_crashed(controller.players[i])

    done = all(crashed)
    pygame.display.update()

    clock.tick(40)

for i in range(genetic.pop_size):
    print(controller.players[i].max_block * controller.players[i].laps)
