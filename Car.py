class Car():
    def __init__(self):
        self.x = 4e-1
        self.y = 3e-1
        self.angle = 0e0
        self.vel = 0e0
        self.block = 0e0
        self.crash = False
        self.acc = 5e-1
        #10 degree in radians
        self.turn = 0.174

    def get_coor(self):
        return self.x,self.y

    def set_coor(self,x,y):
        self.x = x
        self.y = y

