class Car():
    def __init__(self):
        self.x = 5e-1
        self.y = 5e-1
        self.angle = 0e0
        self.vel = 0e0
        self.block = 0e0

    def get_coor(self):
        return self.x,self.y

    def get_vel(self):
        return self.vel

    def set_coor(self,x,y):
        self.x = x
        self.y = y

    def set_vel(self,vel):
        self.vel = vel

    def get_angle(self):
        return self.angle

    def set_angle(self,angle):
        self.angle = angle

    def get_block(self):
        return self.block

    def set_block(self,block):
        self.block = block
