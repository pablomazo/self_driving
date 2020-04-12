import pickle
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

    def save_parameters(self, filename='checkpoint.pickle'):
        parameters = {}
        parameters['w_input'] = self.w_input
        parameters['b_input'] = self.b_input
        parameters['w_output'] = self.w_output
        parameters['b_output'] = self.b_output

        with open(filename, 'wb') as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, filename='checkpoint.pickle'):
        with open(filename, 'rb') as handle:
            parameters = pickle.load(handle)

        self.w_input  =  parameters['w_input']
        self.b_input  =  parameters['b_input']
        self.w_output =  parameters['w_output']
        self.b_output =  parameters['b_output']
