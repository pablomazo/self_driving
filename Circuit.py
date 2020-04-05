class Circuit():
    def __init__(self, circuit_list):
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
        # Height and width of each box.
        self.height, self.width = 60e0, 60e0

        # List to build the circuit.
        self.circuit_list = circuit_list

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

    # TODO: fill method
    def is_wall(self, iblock, point):
        return False
