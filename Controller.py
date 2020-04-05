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
            xlim=x0
            ylim=y0
            if np.cos(angle) >= 0e0:
                xlim = x1
            if np.sin(angle) >= 0e0:
                ylim = y1

            x = xlim 
            y = ylim

            if np.abs(np.sin(angle)) > 1e-5:
                x = np.cos(angle)/np.sin(angle)*\
                        (ylim - car.get_coor()[1]) + car.get_coor()[0]
            else:
                x = xlim
                y = car.get_coor()[1]
    
            if x0 <= x <= x1:
                wall = True
               # wall = my_circuit.is_wall(block,x,y0)
    
            else:
                x = xlim 
                y = ylim
                if np.abs(np.cos(angle)) > 1e-5:
                    y = np.sin(angle)/np.cos(angle)*\
                            (xlim - car.get_coor()[0]) + car.get_coor()[1]
                else:
                    y = car.get_coor[1]

                wall = True
               # wall = my_circuit.is_wall(x0,y)

            block += 1
            #if block ==  my_circuit.nblocks: block = 0
        dist = np.sqrt((x-car.get_coor()[0])**2+(y-car.get_coor()[1])**2)
        return dist


    
     

main_controller = Controller()
car1 = Car()
