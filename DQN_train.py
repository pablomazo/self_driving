from dqn import ReplayMemory
import pygame
from Controller import Controller
import sys

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create Window
WHITE = (255,255,255)
BLUE = (0,0,255)
BLACK = (0,0,0)

screen_size = (0, 0)
pygame.init()
screen = pygame.display.set_mode(screen_size)


#Instanciate Controller
controller = Controller()
controller.load_circuit()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
H1 = 5
N_ACTIONS = 4
memory = ReplayMemory(BATCH_SIZE)

player = DQNPlayer(H1, train = True, device = device)
controller.register_player(player)
player = controller.player[0]

optimizer = optim.RMSprop(player.policy.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def greedy_policy():
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    return eps_threshold

######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model(player):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = player.policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = player.target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in player.policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#


num_episodes = 50
for i_episode in range(num_episodes):
      try:
        print('Episode:', i_episode)
        if i_episode % n_change == 0:
            controller.load_circuit()
        controller.reset()
        reward_1 = 0
        done = False

        clock = pygame.time.Clock()
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            screen.fill(BLACK)

            #Repainting all circuit
            for block in range(controller.circuit.nblocks):
                x0, _, y0, _ = controller.circuit.get_block_coor(block)
                rect = pygame.Rect(x0, y0, controller.circuit.width, controller.circuit.height)

                if block == 0:
                    pygame.draw.rect(screen, BLUE, rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)


            end = player.count >= 100 or player.laps >= 10
            if end: player.crashed = True
            if not player.crashed:
                player.draw(screen)

                state = controller.get_state(player)

                eps_threshold = greedy_policy()
                player.action = select_action(state, N_ACTIONS, eps_threshold)
                key = player.key_from_action(action)
                controller.exec_action(player, key)

                done = player.crashed
                
                if not done:
                    next_state = controller.get_state(player)
                    reward = player.car.block - reward_1

                else:
                    next_state = None
                    reward = -10

                reward_1 = reward
                reward = torch.tensor([reward], device=device)
                state = torch.tensor([state], device=device)
                next_state = torch.tensor([next_state], device=device)


                # Store the transition in memory
                memory.push(state, action, next_state, reward)
         
                # Perform one step of the optimization (on the target network)
                optimize_model(player)
    
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            player.target.load_state_dict(player.policy.state_dict())

    except KeyboardInterrupt:
        torch.save(player.policy.state_dict(), "best_dqn_model.pth")



