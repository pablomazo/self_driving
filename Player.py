import pygame

class Player():
    def __init__(self, car_pos):
        self.GUI_car = pygame.rect.Rect((car_pos[0], car_pos[1], 10, 10))
        self.color = (0,0,255)

    def handle_keys(self):
        key = pygame.key.get_pressed()
        dis = 1
        if key[pygame.K_LEFT]:
           self.GUI_car.move_ip(-dis, 0)
        if key[pygame.K_RIGHT]:
           self.GUI_car.move_ip(dis, 0)
        if key[pygame.K_UP]:
           self.GUI_car.move_ip(0, -dis)
        if key[pygame.K_DOWN]:
           self.GUI_car.move_ip(0, dis)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.GUI_car)
