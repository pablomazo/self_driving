import numpy as np
from Player import SupervisedPlayer
from NeuralNetworks import *
import torch


batch_size = 60000
p = 4

def get_training_set():

#    data = np.loadtxt('input_supervised')
#    input_batch = data[: batch_size, :]
#    input_batch = torch.tensor(input_batch, dtype=torch.float)

    input_batch = torch.rand((batch_size, p))
    output_batch = torch.zeros((batch_size, p), dtype=torch.float)

    for i in range(batch_size):
        input_batch[i][0] *= 1e0 #300
        input_batch[i][1] *= 1e0 #300
        input_batch[i][2] *= 1e0 #300
        input_batch[i][3] *= 1e0 #5.1 * np.random.randint(2)
        output_batch[i][input_batch[i,0:3].max(0)[1]] = 1e0

    return input_batch, output_batch


def train(neural_net, X, Y, optimizer, loss_fun):

    y_pred = neural_net(X)
    loss = loss_fun(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


lr = 0.001
structure = [5]
net = FF1H(structure)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = torch.nn.MSELoss()

input_batch, output_batch = get_training_set()

l2_prev = 1e10
done = False
i = 0
#for i in range(rango):
while not done:
    l2_cost = train(net, input_batch, output_batch, optimizer, loss)

    loss_var = np.abs(l2_cost - l2_prev) / l2_prev

    if i % 100== 0:
        print(i, l2_cost, loss_var)

    done = loss_var < 1e-5
    l2_prev = l2_cost
    i += 1


for i in range(10):
    ind = np.random.randint(batch_size)
    print(input_batch[ind])
    print(output_batch[ind])
    print(net(input_batch[ind].view(1, -1)))
    print("-----------------------------------------------------------------------------")


player = SupervisedPlayer(GUI=False,
                          network_class=net.__class__.__name__,
                          structure=structure)

player.network.load_state_dict(net.state_dict())
player.save_network(filename='supervised.pth')
