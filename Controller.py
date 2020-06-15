import numpy as np
from Car import Car
from Circuit import Circuit
from Player import HeuristicPlayer, GeneticPlayer
from GA_population import GA_population

class Controller():
    '''
    Controller class

    '''

    def __init__(self):
        '''
        instantiate list of players
        '''

        self.players = []


    def load_circuit(self, circuit_id = None):
        '''
        load circuit
        if circuit_id is given the circuit recuested is created
        if no circuit_id is given it will load random circuit 
        from the list of all circuits
        '''

        self.circuit = Circuit(circuit_id)
        self.circuit.build_circuit()
        center =  self.circuit.limits
        x0, x1, y0, y1 = self.circuit.get_block_coor(0)

        #Put initial position of the Player in the center of the block
        self.initial_pos = [(x1-x0)/2e0+x0, (y1-y0)/2e0+y0]

        #Set orientation according to the position of the second block
        orientation_dict = {1: 0e0, 2: -np.pi / 2, 3: np.pi, 4: np.pi / 2}
        self.orientation = orientation_dict[self.circuit.circuit_list[0]]

    
    def reset(self):
        '''
        Set all variables in Player and Car to their initial values
        '''

        for i, player in enumerate(self.players):
            player.reset()
            if player.__class__.__name__ == 'GeneticPlayer':
                player.network = self.genetic.population[i]

            player.car.reset(self.initial_pos)
            player.car.angle = self.orientation

    def register_genetic(self,genetic, GUI=True):
        '''
        Creates and adds a list of genetic players 
        the number of players is given by the population
        This method is used for training GeneticPlayer
        '''

        self.genetic = genetic
        for i in range(self.genetic.pop_size):
            player = GeneticPlayer(self.genetic.population[i], GUI=GUI)
            self.register_player(player)

    def register_player(self, player):
        '''
        Adds the new player to the list players
        '''

        player.car.set_coor(self.initial_pos[0],self.initial_pos[1])
        player.car.angle = self.orientation
        self.players.append(player)

    def car_dist(self,car,desv):
        '''
        Gives the distance between the car and the next wall
            
            Arguments
            - desv is the desviation with respect the
              car angle
        '''

        #Sum the desviation to the car's angle
        angle = car.angle + desv
        block = car.block
        wall = False

        #Iterates until a wall is reached
        while not wall:
            x0, x1, y0, y1 = self.circuit.circuit[block]

            #Saving the limit of the actual block 
            #according to orientation of the car
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

            #If not wall is find go to the next block
            block += 1
            #If the next block is the last one reset block to 0
            if block == self.circuit.nblocks: block = 0

        #Calculate distance between the position of the car and the point of the wall
        dist = np.sqrt((x-car.get_coor()[0])**2+(y-car.get_coor()[1])**2)
        return dist

    def is_out(self,x,y,block):
        '''
        Returns true if the position given by (x,y)
        is out of the limits of the block
        '''

        out = False
        x0, x1, y0, y1 = self.circuit.get_block_coor(block)
        if x1 < x or x < x0 or y1 < y or y < y0:
            out = True
        return out

    def up_button_pressed(self,player):
        '''
        Increase car velocity
        '''

        player.car.vel += player.car.acc
        self.update_position(player)

    def left_button_pressed(self,player):
        '''
        Turn car to the left and decrease car velocity
        '''

        player.car.angle -= player.car.turn
        player.car.vel = np.amax([0e0, player.car.vel - player.car.acc])
        self.update_position(player)

    def right_button_pressed(self,player):
        '''
        Turn car to the right and decrease car velocity
        '''

        player.car.angle += player.car.turn
        player.car.vel = np.amax([0e0, player.car.vel - player.car.acc])
        self.update_position(player)

    def none_button_pressed(self,player):
        '''
        Decrease car velocity
        '''

        player.car.vel = np.amax([0e0, player.car.vel - player.car.acc])
        self.update_position(player)

    def update_position(self,player):
        '''
        Calculates new car's position
        if car reaches a wall velocity is set to 0
        '''
        x,y = player.car.get_coor()
        x += player.car.vel * np.cos(player.car.angle)
        y += player.car.vel * np.sin(player.car.angle)
        player.count += 1
        if self.is_out(x,y,player.car.block):
            newblock = player.car.block + 1
            if newblock == self.circuit.nblocks: newblock = 0
            if self.is_out(x,y,newblock):
                player.car.vel = 0e0
                player.crashed = True
                x,y = player.car.get_coor()
            else:
                player.count = 0
                player.car.block += 1
                if player.car.block == self.circuit.nblocks:
                    player.car.block = 0
                    player.laps += 1

        player.car.set_coor(x,y)

    def exec_action(self, player, key):
        '''
        Actualizes player's state according to the action selected by the player
        '''

        if key == 'U':
            self.up_button_pressed(player)

        elif key == 'L':
            self.left_button_pressed(player)

        elif key == 'R':
            self.right_button_pressed(player)

        elif key == None:
            self.none_button_pressed(player)

    def get_state(self, player):
        '''
        Returns the distances of the car to the next walls 
        and the car velocity

            - Angles of the rays (-45º, 0º, 45º)
        '''

        # Angles (-45º, 0º, 45º)
        angles = [np.pi / 4e0, 0e0, - np.pi / 4e0]

        state = []
        for angle in angles:
            dis = self.car_dist(player.car, angle)
            state.append(dis)

        state += [player.car.vel]

        return state

    def set_state(self, player):
        '''
        Saves distances of the car to the next walls
        and the car velocity in Player's state variable

            - Angles of the rays (-45º, 0º, 45º)
        '''
        # Angles (-45º, 0, 45º)
        angles = [np.pi / 4e0, 0e0, - np.pi / 4e0]

        state = []
        for angle in angles:
            dis = self.car_dist(player.car, angle)
            state.append(dis)

        player.state = state
