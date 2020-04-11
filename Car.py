class Car():
    def __init__(self):
        self.x = 4e-1
        self.y = 3e-1
        self.angle = 0e0
        self.vel = 0e0
        self.block = 0
        self.acc = 1e-1
        #10 degree in radians
        self.turn = 0.104

    def get_coor(self):
        return self.x,self.y

    def set_coor(self,x,y):
        self.x = x
        self.y = y

