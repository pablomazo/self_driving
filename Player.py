import pygame
import numpy as np
import NeuralNetwork as NeuralNetwork

class Player():
    def __init__(self):
        self.GUI_car_orig = pygame.image.load('./images/car2.png').convert_alpha()
        self.GUI_car_orig = pygame.transform.scale(self.GUI_car_orig, (14, 24))
        self.GUI_car_orig = pygame.transform.rotate(self.GUI_car_orig, -90)
        self.GUI_car = self.GUI_car_orig.copy()

    def register_car(self, a_car):
        self.car = a_car

    def handle_keys(self):
        key = pygame.key.get_pressed()

        if key[pygame.K_LEFT]:
            return 'L'

        if key[pygame.K_RIGHT]:
            return 'R'

        if key[pygame.K_UP]:
            return 'U'

    def draw(self, surface):
        angle = np.degrees(-self.car.angle)
        x,y = self.car.get_coor()
        x -= 7
        y -= 12
        coor = x, y
        self.GUI_car = pygame.transform.rotate(self.GUI_car_orig, angle)
        surface.blit(self.GUI_car, coor)

class Player2():
    def __init__(self):
        self.GUI_car_orig = pygame.image.load('./images/car.png').convert_alpha()
        self.GUI_car_orig = pygame.transform.scale(self.GUI_car_orig, (14, 24))
        self.GUI_car_orig = pygame.transform.rotate(self.GUI_car_orig, -90)
        self.GUI_car = self.GUI_car_orig.copy()

    def register_car(self, a_car):
        self.car = a_car

    def handle_keys(self):
        key = pygame.key.get_pressed()

        if key[pygame.K_a]:
            return 'L'

        if key[pygame.K_d]:
            return 'R'

        if key[pygame.K_w]:
            return 'U'

    def draw(self, surface):
        angle = np.degrees(-self.car.angle)
        x,y = self.car.get_coor()
        x -= 7
        y -= 12
        coor = x, y
        self.GUI_car = pygame.transform.rotate(self.GUI_car_orig, angle)
        surface.blit(self.GUI_car, coor)

class HeuristicPlayer():
    def __init__(self):
        self.GUI_car_orig = pygame.image.load('./images/car.png').convert_alpha()
        self.GUI_car_orig = pygame.transform.scale(self.GUI_car_orig, (14, 24))
        self.GUI_car_orig = pygame.transform.rotate(self.GUI_car_orig, -90)
        self.GUI_car = self.GUI_car_orig.copy()

        self.state = []

    def register_car(self, a_car):
        self.car = a_car

    def get_key(self):
        keys = ['R', 'U', 'L']

        key_id = np.argmax(self.state)

        return keys[key_id]

    def handle_keys(self):
        key = self.get_key()

        return key


    def draw(self, surface):
        angle = np.degrees(-self.car.angle)
        x,y = self.car.get_coor()
        x -= 7
        y -= 12
        coor = x, y
        self.GUI_car = pygame.transform.rotate(self.GUI_car_orig, angle)
        surface.blit(self.GUI_car, coor)

class GeneticPlayer():
    def __init__(self, network):
        self.GUI_car_orig = pygame.image.load('./images/car.png').convert_alpha()
        self.GUI_car_orig = pygame.transform.scale(self.GUI_car_orig, (14, 24))
        self.GUI_car_orig = pygame.transform.rotate(self.GUI_car_orig, -90)
        self.GUI_car = self.GUI_car_orig.copy()

        self.state = []

        self.network = network

        self.crashed = False

        self.max_block = 0

        self.count = 0

        self.laps = 1

    def reset(self):
        self.crashed = False
        self.max_block = 0
        self.count = 0
        self.laps = 1

    def register_car(self, a_car):
        self.car = a_car

    def get_key(self):
        keys = ['R', 'U', 'L']

        new_state = np.append(self.state,self.car.vel)
        output = self.network.forward(new_state)
        key_id = np.argmax(output)

        return keys[key_id]

    def handle_keys(self):
        key = self.get_key()

        return key


    def draw(self, surface):
        angle = np.degrees(-self.car.angle)
        x,y = self.car.get_coor()
        x -= 7
        y -= 12
        coor = x, y
        self.GUI_car = pygame.transform.rotate(self.GUI_car_orig, angle)
        surface.blit(self.GUI_car, coor)
