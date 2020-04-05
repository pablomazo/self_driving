import pygame
import numpy as np
from Circuit import Circuit
from Player import Player
from Car import Car
from Controller import Controller

WHITE = (255,255,255)
BLUE = (0,0,255)
BLACK = (0,0,0)


screen_size = (0, 0)
pygame.init()
screen = pygame.display.set_mode(screen_size)
done = False

#Instanciate Controller
controller = Controller()

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

    #Repainting car
    controller.player1.draw(screen)

    key = controller.player1.handle_keys()
    if key == 'U':
        controller.up_button_pressed(controller.player1)

    elif key == 'L':
        controller.left_button_pressed(controller.player1)

    elif key == 'R':
        controller.right_button_pressed(controller.player1)

    elif key == None:
        controller.none_button_pressed(controller.player1)


    pygame.display.update()

    clock.tick(40)
