import pygame
import numpy as np
from Circuit import Circuit
from Player import Player
from Car import Car

WHITE = (255,255,255)
BLUE = (0,0,255)
BLACK = (0,0,0)

# Instanciate circuit.
circuit_list = [1,1,1,1,1,1,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4]
circuit = Circuit(circuit_list)

# Build circuit.
circuit.build_circuit()

screen_size = (1000, 500)
pygame.init()
screen = pygame.display.set_mode(screen_size)
done = False

center = circuit.limits
print('center:',center)

x0, _, _, y1 = circuit.get_block_coor(0)
x0 -= center[0]
y1 -= center[1]

player1 = Player()
car1 = Car()
x0, y1 = car1.get_coor()
car1.set_coor(x0 - center[0], y1 - center[1])

player1.register_car(car1)
step_vel = 5e-1
step_angle = np.radians(10e0)

clock = pygame.time.Clock()
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill(BLACK)

    for block in range(circuit.nblocks):
        x0, _, _, y1 = circuit.get_block_coor(block)
        x0 -= center[0]
        y1 -= center[1]
        rect = pygame.Rect(x0, y1, circuit.width, circuit.height)
        pygame.draw.rect(screen, WHITE, rect)

    player1.draw(screen)
    key = player1.handle_keys()
    if key == 'U':
        x0, y1 = player1.car.get_coor()
        player1.car.vel += step_vel
        x0 += player1.car.vel * np.cos(player1.car.angle)
        y1 += player1.car.vel * np.sin(player1.car.angle)

        player1.car.set_coor(x0,y1)
        print('pos', player1.car.x, player1.car.y)
        print('vel', player1.car.vel)
        print('angle:',player1.car.angle)
        print('--------------------------------------')

    elif key == 'L':
        x0, y1 = player1.car.get_coor()
        player1.car.angle -= step_angle
        x0 += player1.car.vel * np.cos(player1.car.angle)
        y1 += player1.car.vel * np.sin(player1.car.angle)
        player1.car.set_coor(x0,y1)
        print(player1.car.angle)

    elif key == 'R':
        x0, y1 = player1.car.get_coor()
        player1.car.angle += step_angle
        x0 += player1.car.vel * np.cos(player1.car.angle)
        y1 += player1.car.vel * np.sin(player1.car.angle)
        player1.car.set_coor(x0,y1)
        print('pos', player1.car.x, player1.car.y)
        print('vel', player1.car.vel)
        print('angle:',player1.car.angle)
        print('--------------------------------------')

    elif key == None:
        player1.car.vel = np.amax([0e0, player1.car.vel - step_vel])



    pygame.display.update()

    clock.tick(40)
