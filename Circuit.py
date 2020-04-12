import numpy as np
class Circuit():
    def __init__(self, circuit_id = None):
        '''
        circuit_list tells how the circuit is built.
        The first box will have a fixed position. Then the indexes
        of the list will tell to which side the next box is join.

                 4
       y1      ----
             3 |  | 1
       y0      ----
                 2
              x0   x1
        '''

        circuits =[ [1,1,1,1,1,1,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,4], [1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1],[1, 4, 4, 4, 4, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],[2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],[3, 2, 2, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1, 4, 4, 3, 3, 4, 4, 3, 3]]

        # Height and width of each box.
        self.height, self.width = 60e0, 60e0

        # List to build the circuit.
        n_circuits = len(circuits)
        if circuit_id == None or circuit_id >= n_circuits: circuit_id = np.random.randint(n_circuits)
        self.circuit_list = circuits[circuit_id]

        # Count number of elements:
        min_val = 1000000000
        for elem in range(1,5):
            tmp_elem = self.circuit_list.count(elem)
            if tmp_elem < min_val:
                self.last = elem
                min_val = tmp_elem

        # Number of blocks in circuit:
        self.nblocks = len(self.circuit_list) + 1

        # Register minimum values of x0 and minimum of y0. 
        self.limits = [0e0, 0e0]

    def build_circuit(self):
        '''
        Will generate a list of x0, x1, y0, y1 positions for each
        box
        '''
        self.circuit = [[0e0, self.width, 0e0, self.height]]

        for box in self.circuit_list:
            prev = self.circuit[-1]
            if box == 1:
                pos = [prev[1], prev[1] + self.width,
                       prev[2], prev[3]]

            elif box == 2:
                pos = [prev[0], prev[1],
                       prev[2] - self.height, prev[2]]

            elif box == 3:
                pos = [prev[0] - self.width, prev[0],
                       prev[2], prev[3]]

            elif box == 4:
                pos = [prev[0], prev[1],
                       prev[3], prev[3] + self.height]

            if pos[0] < self.limits[0]: self.limits[0] = pos[0]
            if pos[2] < self.limits[1]: self.limits[1] = pos[2]
            self.circuit.append(pos)

        # Translate block to fit on screen:
        self.translate_circuit()

    def translate_circuit(self):
        for block in range(self.nblocks):
            self.circuit[block][0] -= self.limits[0]
            self.circuit[block][1] -= self.limits[0]

            self.circuit[block][2] -= self.limits[1]
            self.circuit[block][3] -= self.limits[1]

    def get_block_coor(self, iblock):
        return self.circuit[iblock]

    def is_wall(self, x, y, iblock):
        '''
        Checks if a point x, y is a wall in the circuit.
        '''

        # Get where the wall are for this block:
        wall = [1,2,3,4]

        if iblock == self.nblocks - 1:
            wall.remove(self.last)
        else:
            wall.remove(self.circuit_list[iblock])


        for elem in wall:
            if elem == 1 and abs(x - self.circuit[iblock][1]) < 1e-5:
                return True
            elif elem == 2 and abs(y - self.circuit[iblock][2]) < 1e-5:
                return True
            elif elem == 3 and abs(x - self.circuit[iblock][0]) < 1e-5:
                return True
            elif elem == 4 and abs(y - self.circuit[iblock][3]) < 1e-5:
                return True

        return False
