import numpy as np
from Car import Car
#from Circuit import Circuit

class Controller():

    def __init__(self):
        print("Se crea controlador")
        #my_circuit = Circuit()

    def car_dist(self,car,desv):
        angle = car.get_angle() + desv
        block = car.get_block()
        wall = False

        while not wall:
            x0, x1, y0, y1 = 0e0, 1e0, 0e0, 1e0
           # x0, x1, y0, y1 = my_circuit.get_block_coor(block)
            if np.cos(angle) >= 0e0:
                x0 = x1
            if np.sin(angle) >= 0e0:
                y0 = y1

            x = car.get_coor()[0]
            y = car.get_coor()[1]

            if np.sin(angle) != 0e0:
                x = np.cos(angle)/np.sin(angle)*\
                        (x0 - car.get_coor()[0]) + car.get_coor()[1]
            else:
                x = x0
    
            if x0 <= x <= x1:
                wall = True
               # wall = my_circuit.is_wall(block,x,y0)
    
            else:
                if np.cos(angle) != 0e0:
                    y = np.sin(angle)/np.cos(angle)*\
                            (y0 - car.get_coor()[1]) + car.get_coor()[0]
                else:
                    y = y0

                wall = True
               # wall = my_circuit.is_wall(x0,y)

            block += 1
            #if block ==  my_circuit.nblocks: block = 0

        print(x," ",car.get_coor()[0]," ",y," ",car.get_coor()[1])
        dist = np.sqrt((x-car.get_coor()[0])**2+(y-car.get_coor()[1])**2)
        return dist
    
     

main_controller = Controller()
car1 = Car()
print(main_controller.car_dist(car1,0e0))
