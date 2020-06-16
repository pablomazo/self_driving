import pygame
import numpy as np
from Controller import Controller
from Player import *
from NeuralNetworks import *
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
controller.load_circuit(1)
player = HumanPlayer1()
player2 = HumanPlayer2()
player3 = HeuristicPlayer()

# Supervised player.
player4 = NeuralPlayer(model_file='./saved_models/final_supervised.pth')

# Genetic Player.
player5 = NeuralPlayer(model_file='./saved_models/best_genetic.pth', color='red')

# DQN player.
player6 = DQNPlayer(model_file='./saved_models/final_DQN.pth')

controller.register_player(player)
controller.register_player(player2)
controller.register_player(player3)
controller.register_player(player4)
controller.register_player(player5)
controller.register_player(player6)

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
