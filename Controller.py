import numpy as np
from Car import Car
from Circuit import Circuit
from Player import Player, Player2, HeuristicPlayer, GeneticPlayer
from GA_population import GA_population

class Controller():

    def __init__(self):
        #Instanciate circuit
        #circuit_list = [1,1,1,1,1,1,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,4]
        circuit_list = [1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1]
        self.circuit = Circuit(circuit_list)

        #Build circuit
        self.circuit.build_circuit()

        center = self.circuit.limits

        #self.player2 = HeuristicPlayer()
        #self.player1 = Player()
        #self.player2 = Player2()

        #car1 = Car()
        #car2 = Car()


        #car1.set_coor((x1-x0)/2e0+x0, (y1-y0)/2e0+y0)
        #car2.set_coor((x1-x0)/2e0+x0, (y1-y0)/2e0+y0)

        #self.player1.register_car(car1)
        #self.player2.register_car(car2)

    def register_genetic(self,genetic):
        self.genetic = genetic

    def initialize_genetic_players(self):
        self.players = []
        x0, x1, y0, y1 = self.circuit.get_block_coor(0)
        for i in range(self.genetic.pop_size):
            player = GeneticPlayer(self.genetic.population[i])
            car = Car()
            car.set_coor((x1-x0)/2e0+x0, (y1-y0)/2e0+y0)
            player.register_car(car)
            self.players.append(player)


    def car_dist(self,car,desv):
        angle = car.angle + desv
        block = car.block
        wall = False

        while not wall:
            x0, x1, y0, y1 = self.circuit.circuit[block]

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
                wall = self.circuit.is_wall(x,y,block)

            else:
                x = xlim
                y = ylim
                if np.abs(np.cos(angle)) > 1e-5:
                    y = np.sin(angle)/np.cos(angle)*\
                            (xlim - car.get_coor()[0]) + car.get_coor()[1]
                else:
                    y = car.get_coor[1]

                wall = self.circuit.is_wall(x,y,block)

            block += 1
            if block == self.circuit.nblocks: block = 0

        dist = np.sqrt((x-car.get_coor()[0])**2+(y-car.get_coor()[1])**2)
        return dist

    def is_out(self,x,y,block):
        out = False
        x0, x1, y0, y1 = self.circuit.get_block_coor(block)
        if x1 <= x or x <= x0 or y1 <= y or y <= y0:
            out = True
        return out

    def up_button_pressed(self,player):
        player.car.vel += player.car.acc
        self.update_position(player)

    def left_button_pressed(self,player):
        player.car.angle -= player.car.turn
        player.car.vel = np.amax([0e0, player.car.vel - player.car.acc])
        self.update_position(player)

    def right_button_pressed(self,player):
        player.car.angle += player.car.turn
        player.car.vel = np.amax([0e0, player.car.vel - player.car.acc])
        self.update_position(player)

    def none_button_pressed(self,player):
        player.car.vel = np.amax([0e0, player.car.vel - player.car.acc])
        self.update_position(player)

    def update_position(self,player):
        x,y = player.car.get_coor()
        x += player.car.vel * np.cos(player.car.angle)
        y += player.car.vel * np.sin(player.car.angle)
        player.count += 1
        if self.is_out(x,y,player.car.block):
            newblock = player.car.block + 1
            if newblock == self.circuit.nblocks: newblock = 0
            if self.is_out(x,y,newblock):
                player.car.vel = 0e0
                x,y = player.car.get_coor()
            else:
                player.count = 0
                player.car.block += 1
                if player.car.block == self.circuit.nblocks: 
                    player.car.block = 0

        player.car.set_coor(x,y)

    def exec_action(self, player, key):
        if key == 'U':
            self.up_button_pressed(player)

        elif key == 'L':
            self.left_button_pressed(player)

        elif key == 'R':
            self.right_button_pressed(player)

        elif key == None:
            self.none_button_pressed(player)

    def set_state(self, player):
        # Angles
        angles = [np.pi / 4e0, 0e0, - np.pi / 4e0]

        state = []
        for angle in angles:
            dis = self.car_dist(player.car, angle)
            state.append(dis)

        player.state = state

    def is_crashed(self, player):
        if self.car_dist(player.car,0e0) < 1e-1:
            player.crashed = True
            player.max_block = player.car.block

