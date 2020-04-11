import numpy as np
class NeuralNetwork():
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.w_input = np.random.rand(n_inputs,n_hidden)
        self.b_input = np.random.rand(n_hidden)
        self.w_output = np.random.rand(n_hidden, n_outputs)
        self.b_output = np.random.rand(n_outputs)

    def propagate(self, inputs, w, b):
        h = inputs.dot(w)
        h += b
        return self.sigmoid(h)

    def sigmoid(self, activation):
        return 1.0e0 / (1.0e0 + np.exp(-activation))

    def forward(self, inputs):
        h1 = self.propagate(inputs, self.w_input, self.b_input)
        return self.propagate(h1, self.w_output, self.b_output)

NN = NeuralNetwork(2,3,4)
print(NN.forward(NN.input))
