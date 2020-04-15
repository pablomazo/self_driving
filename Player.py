import pygame
from abc import ABC, abstractmethod
import numpy as np
import NeuralNetwork as NeuralNetwork
from Car import Car

class Player(ABC):
    def __init__(self):
        self.count = 0
        self.laps = 0
        self.crashed = False
        self.car = Car()

    def set_image(self, image_name):
        self.GUI_car_orig = pygame.image.load(image_name).convert_alpha()
        self.GUI_car_orig = pygame.transform.scale(self.GUI_car_orig, (14, 24))
        self.GUI_car_orig = pygame.transform.rotate(self.GUI_car_orig, -90)
        self.GUI_car = self.GUI_car_orig.copy()

    def register_car(self, a_car):
        self.car = a_car

    def handle_keys(self):
        key = self.get_key()
        return key

    @abstractmethod
    def get_key(self):
        pass

    def draw(self, surface):
        angle = np.degrees(-self.car.angle)
        x,y = self.car.get_coor()
        x -= 7
        y -= 12
        coor = x, y
        self.GUI_car = pygame.transform.rotate(self.GUI_car_orig, angle)
        surface.blit(self.GUI_car, coor)

    def reset(self):
        self.crashed = False
        self.count = 0
        self.laps = 0

class HumanPlayer1(Player):
    def __init__(self):
        super().__init__()
        self.set_image('./images/car.png')

    def get_key(self):
        key = pygame.key.get_pressed()

        if key[pygame.K_LEFT]:
            return 'L'

        if key[pygame.K_RIGHT]:
            return 'R'

        if key[pygame.K_UP]:
            return 'U'

class HumanPlayer2(Player):
    def __init__(self):
        super().__init__()
        self.set_image('./images/car2.png')

    def get_key(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_a]:
            return 'L'

        if key[pygame.K_d]:
            return 'R'

        if key[pygame.K_w]:
            return 'U'

class HeuristicPlayer(Player):
    def __init__(self):
        super().__init__()
        self.set_image('./images/car3.png')

    def get_key(self):
        keys = ['R', 'U', 'L']

        key_id = np.argmax(self.state)
        if self.state[1] <= 1e-1 :key_id = 2

        return keys[key_id]

class GeneticPlayer(Player):
    def __init__(self, network):
        super().__init__()
        self.set_image('./images/car4.png')

        self.state = []

        self.network = network

    def get_key(self):
        keys = ['R', 'U', 'L', None]

        new_state = np.append(self.state,self.car.vel)
        output = self.network.forward(new_state)
        key_id = np.argmax(output)

        return keys[key_id]
