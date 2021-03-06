import pygame
from abc import ABC, abstractmethod
import numpy as np
from Car import Car
from NeuralNetworks import *
import random
import torch
import importlib

class Player(ABC):
    def __init__(self):
        self.count = 0
        self.laps = 0
        self.crashed = False
        self.car = Car()

    def set_image(self, color):
        image_name = './images/{}.png'.format(color)
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
    def __init__(self, color='black'):
        super().__init__()
        self.set_image(color)

    def get_key(self):
        key = pygame.key.get_pressed()

        if key[pygame.K_LEFT]:
            return 'L'

        if key[pygame.K_RIGHT]:
            return 'R'

        if key[pygame.K_UP]:
            return 'U'

class HumanPlayer2(Player):
    def __init__(self, color='green'):
        super().__init__()
        self.set_image(color)

    def get_key(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_a]:
            return 'L'

        if key[pygame.K_d]:
            return 'R'

        if key[pygame.K_w]:
            return 'U'

class HeuristicPlayer(Player):
    def __init__(self, color='yellow'):
        super().__init__()
        self.set_image(color)

    def get_key(self):
        keys = ['R', 'U', 'L']

        key_id = np.argmax(self.state)
        if self.state[1] <= 1e-1 :key_id = 2

        return keys[key_id]

class GeneticPlayer(Player):
    def __init__(self, network, GUI=True, color='red'):
        super().__init__()

        if GUI:
            self.set_image(color)

        self.state = []

        self.network = network

    def get_key(self):
        keys = ['R', 'U', 'L', None]

        with torch.no_grad():
            new_state = torch.tensor(np.append(self.state,self.car.vel), dtype=torch.float)
            output = self.network(new_state)
            key_id = output.max(0)[1]

        return keys[key_id]

class DQNPlayer(Player):
    '''
    Class for DQN player.

    Arguments:
        - network_class: Name of the NN class to be used.
        - structure: List defining the structure of the NN.
        - device: 'cpu' or 'cuda' to evaluate the model in CPU or GPU.
        - GUI: True of False to load image of the car to be used on a GUI.
        - model_file: Path to a model file to be loaded on nets.
        If a model_file is specified, the network_class and structure
        will be those defined in the model file.
        - train: If True, a target net is initialized.
        - color: Color the car will have.

    '''
    def __init__(self, network_class='FF2H_sigmoid',
                       structure=[5,5],
                       device='cpu',
                       GUI=True,
                       model_file=None,
                       train=False,
                       color='blue'):
        super().__init__()

        self.device = device
        self.state = []
        self.network_class = network_class
        self.train = train

        if GUI:
            self.set_image(color)

        # If model file is given, read structure and NN class.
        if model_file is not None:
            model_dict = torch.load(model_file, map_location=self.device)
            structure = model_dict['structure']
            self.network_class = model_dict['model_class']

            print('Using model file: {}'.format(model_file))
            print('network class: {}'.format(self.network_class))
            print('Structure: {}'.format(structure))

        model_class = getattr(importlib.import_module("NeuralNetworks"), self.network_class)

        self.policy = model_class(structure).to(device)

        # If a model file is given, load parameters into net.
        if model_file is not None:
            self.policy.load_state_dict(model_dict["state_dict"])

        # A target net is inialized with the same parameters as policy.
        if self.train:
            self.target = model_class(structure).to(device)
            self.target.load_state_dict(self.policy.state_dict())
            self.target.eval()
        else:
            self.policy.eval()

    def get_key(self):
        keys = ['R', 'U', 'L', None]

        new_state = np.append(self.state,self.car.vel)
        key_id, _ = self.select_action(new_state)

        return keys[key_id]

    def key_from_action(self, action):
        keys = ['R', 'U', 'L', None]

        return keys[action]

    def select_action(self, state, n_actions=None, eps_threshold=0e0):
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float)
                action = self.policy(state).max(1)[1].view(1,1)
                Q = self.policy(state).max(1)[0]
                return action, Q
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long), None

    def save_network(self, filename='DQN_network.pth'):
        tosave = {}
        tosave['structure'] = self.network.structure
        tosave['state_dict'] = self.network.state_dict()
        torch.save(tosave, filename)


class NeuralPlayer(Player):
    '''
    Class for Supervised  and Genetic player.

    Arguments:
        - network_class: Name of the NN class to be used.
        - structure: List defining the structure of the NN.
        - device: 'cpu' or 'cuda' to evaluate the model in CPU or GPU.
        - GUI: True of False to load image of the car to be used on a GUI.
        - model_file: Path to a model file to be loaded on nets.
        If a model_file is specified, the network_class and structure
        will be those defined in the model file.
        - color: Color the car will have.

    '''

    def __init__(self, network_class='FF1H',
                       structure = [5],
                       device = 'cpu',
                       GUI=True,
                       model_file=None,
                       color='grey'):

        super().__init__()

        self.device = device
        self.state = []
        self.network_class = network_class

        if GUI:
            self.set_image(color)

        if model_file is not None:
            model_info = torch.load(model_file, map_location=self.device)
            structure = model_info['structure']
            self.network_class = model_info['model_class']

            print('Using model file: {}'.format(model_file))
            print('network class: {}'.format(self.network_class))
            print('Structure: {}'.format(structure))

        model_class = getattr(importlib.import_module("NeuralNetworks"), self.network_class)

        self.network = model_class(structure).to(device)

        if model_file is not None:
            self.network.load_state_dict(model_info["state_dict"])

    def get_key(self):
        keys = ['R', 'U', 'L', None]

        new_state = np.append(self.state,self.car.vel)
        expec = np.argmax(self.state)
        key_id, _ = self.select_action(new_state)

        return keys[key_id]

    def key_from_action(self, action):
        keys = ['R', 'U', 'L', None]

        return keys[action]

    def select_action(self, state):

        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float)
            sal = self.network(state)
            action = sal.max(1)[1].view(1,1)
            Q = self.network(state).max(1)[0]
            return action, Q
