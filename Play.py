from GA_population import GA_population
import pygame
import numpy as np
from Controller import Controller
from Player import *
from NeuralNetworks import FF1H, FF2H_relu, FF2H_sigmoid
import torch
import sys

#Create Window
WHITE = (255,255,255)
BLUE = (0,0,255)
BLACK = (0,0,0)

screen_size = (0, 0)
pygame.init()
screen = pygame.display.set_mode(screen_size)

#Instanciate Controller
controller = Controller()
controller.load_circuit(4)
player = HumanPlayer1()
player2 = HumanPlayer2()
player3 = HeuristicPlayer()

player4 = SupervisedPlayer(model_file="supervised.pth")

#net = FF2H_sigmoid(300, 300)
#net.load_parameters_supervised('./supervised.pickle')
#player4 = GeneticPlayer(net)

#net = FF1H(5)
#net.load_parameters('./saved_models/master2.pickle')
#player5 = GeneticPlayer(net)

controller.register_player(player)
controller.register_player(player2)
controller.register_player(player3)
controller.register_player(player4)
#controller.register_player(player5)

done = False

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

    for player in controller.players:
        #Repainting car
        player.draw(screen)

        # Get keys pressed by players.
        controller.set_state(player)
        key = player.handle_keys()
        controller.exec_action(player, key)
        #controller.is_crashed(player)

    pygame.display.update()

    clock.tick(40)
