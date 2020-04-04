import pygame
from Circuit import Circuit
from Player import Player

WHITE = (255,255,255)
BLUE = (0,0,255)
BLACK = (0,0,0)

# Instanciate circuit.
circuit_list = [1,1,1,1,1,1,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4]
circuit = Circuit(circuit_list)

# Build circuit.
circuit.build_circuit()

screen_size = (500,300)
pygame.init()
screen = pygame.display.set_mode(screen_size)
done = False

center = [screen_size[0] / 2e0, screen_size[1] / 2e0]

x0, _, _, y1 = circuit.get_block_coor(0)
x0 += center[0]
y1 += center[1]

player1 = Player([x0,y1])

clock = pygame.time.Clock()
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill(BLACK)

    for block in range(circuit.nblocks):
        x0, _, _, y1 = circuit.get_block_coor(block)
        x0 += center[0]
        y1 += center[1]
        pygame.draw.rect(screen, WHITE, pygame.Rect(x0, y1, 20e0, 20e0))

    player1.draw(screen)
    player1.handle_keys()
    pygame.display.update()

    clock.tick(40)
