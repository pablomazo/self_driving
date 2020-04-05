import pygame
import numpy as np

class Player():
    def __init__(self):
        #self.GUI_car = pygame.rect.Rect((car_pos[0], car_pos[1], 10, 10))
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
            self.car_pos[0] += dis

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
