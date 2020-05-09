import torch
import torch.nn as nn
import torch.nn.functional as F


class FF1H(nn.Module):

    def __init__(self, h):
        super(FF1H, self).__init__()
        h1 = h[0]
        self.structure = [h1]
        self.linear1 = nn.Linear(4, h1)
        self.linear2 = nn.Linear(h1, 4)
    
    def forward(self, x): 
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

class FF2H_relu(nn.Module):

    def __init__(self, h):
        super(FF2H_relu, self).__init__()
        h1, h2 = h[0], h[1]
        self.structure = [h1,h2]
        self.linear1 = nn.Linear(4, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, 4)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class FF2H_sigmoid(nn.Module):

    def __init__(self, h):
        super(FF2H_sigmoid, self).__init__()
        h1, h2 = h[0], h[1]
        self.structure = [h1,h2]
        self.linear1 = nn.Linear(4, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, 4)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x



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

    def load_parameters_supervised(self, filename='supervised.pickle'):
        with open(filename, 'rb') as handle:
            parameters = pickle.load(handle)

        self.w_input = parameters["weight0"]
        self.b_input = parameters["bias0"][0, :]
        self.w_output = parameters["weight1"]
        self.b_output = parameters["bias1"][0, :]


class TwoNeuralNetwork():
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs):
        self.w_input = np.random.rand(n_inputs,n_hidden1)
        self.b_input = np.random.rand(n_hidden1)
        self.w_hidden = np.random.rand(n_hidden1, n_hidden2)
        self.b_hidden = np.random.rand(n_hidden2)
        self.w_output = np.random.rand(n_hidden2, n_outputs)
        self.b_output = np.random.rand(n_outputs)

    def propagate(self, inputs, w, b):
        h = inputs.dot(w)
        h += b
        return self.sigmoid(h)

    def sigmoid(self, activation):
        return 1.0e0 / (1.0e0 + np.exp(-activation))

    def forward(self, inputs):
        h1 = self.propagate(inputs, self.w_input, self.b_input)
        h1 = self.propagate(h1, self.w_hidden, self.b_hidden)
        return self.propagate(h1, self.w_output, self.b_output)

    def save_parameters(self, filename='checkpoint.pickle'):
        parameters = {}
        parameters['w_input'] = self.w_input
        parameters['b_input'] = self.b_input
        parameters['w_output'] = self.w_output
        parameters['b_output'] = self.b_output

        with open(filename, 'wb') as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters_supervised(self, filename='supervised.pickle'):
        with open(filename, 'rb') as handle:
            parameters = pickle.load(handle)

        self.w_input = parameters["weight0"]
        self.b_input = parameters["bias0"][0, :]
        self.w_hidden = parameters["weight1"]
        self.b_hidden = parameters["bias1"][0, :]
        self.w_output = parameters["weight2"]
        self.b_output = parameters["bias2"][0, :]
