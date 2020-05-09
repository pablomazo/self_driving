import numpy as np
from Player import SupervisedPlayer
from NeuralNetworks import FF2H_sigmoid, FF2H_relu
import torch


batch_size = 10000
p = 4

def get_training_set():

    input_batch = torch.rand((batch_size, p))
    output_batch = torch.rand((batch_size, p))
    
    for i in range(batch_size):
        input_batch[i][0] *= 200
        input_batch[i][1] *= 200
        input_batch[i][2] *= 200
        input_batch[i][3] *= 5.1 * np.random.randint(2)
        output_batch[i][input_batch[i,0:3].max(0)[1]] = 1

    return input_batch, output_batch
    


def train(neural_net, X, Y, optimizer, loss_fun):

    y_pred = neural_net(X)
    loss = loss_fun(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


lr = 0.001
net = FF2H_sigmoid([300, 300])
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = torch.nn.MSELoss()

input_batch, output_batch = get_training_set()

rango = 1000
prev_der = None
for i in range(rango):
    l2_cost = train(net, input_batch, output_batch, optimizer, loss)

    if i % 100== 0:
        print(l2_cost, "iteración ",i)


for i in range(1):
    print(input_batch[i])
    print(output_batch[i])
    print(net(input_batch[i].view(1, -1)))
    print("-----------------------------------------------------------------------------")


player = SupervisedPlayer(GUI=False)
player.save_network(filename='supervised.pth')
