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
        self.height, self.width = 20e0, 20e0

        # List to build the circuit.
        self.circuit_list = circuit_list

        # Number of blocks in circuit:
        self.nblocks = len(self.circuit_list) + 1

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

            self.circuit.append(pos)

    def get_block_coor(self, iblock):
        return self.circuit[iblock]

    # TODO: fill method
    def is_wall(self, iblock, point):
        return False
