import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt

from Controller import Controller
from Player import DQNPlayer

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(reward_list, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def greedy_policy(episode, EPS_START, EPS_END, EPS_DECAY):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * episode / EPS_DECAY)
    return eps_threshold

def optimize(agent, optimizer, memory, BATCH_SIZE, GAMMA):
    if len(memory) < BATCH_SIZE:
        return

    # Sample transition:
    sample = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*sample))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    next_state_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)

    #print('State batch:', state_batch)
    #print('Next state batch:', next_state_batch)
    #print('Action batch:', action_batch)
    #print('Reward batch:', reward_batch)

    # End state mask:
    end_state_mask = torch.zeros_like(reward_batch)
    for elem in range(BATCH_SIZE):
        equal = all(state_batch[elem,:] == next_state_batch[elem,:])
        equal = torch.tensor(equal)
        end_state_mask[elem] = torch.where(equal, torch.tensor(0), torch.tensor(1))
    #print('End mask:', end_state_mask)

    # Evaluate policy for "state":
    policy_values = agent.policy(state_batch)
    Q_policy = policy_values.gather(1,action_batch)

    #print('policy values', policy_values)
    #print('Q policy:', Q_policy)

    # Evaluate expected value = reward + gamma * max(target(next_state)) * end_state_mask
    y = agent.target(next_state_batch).max(1)[0].detach()
    y = reward_batch + GAMMA * y * end_state_mask
    y = y.view(-1,1)
    #print('y=', y)

    # Evaluate loss: 
    loss = F.mse_loss(Q_policy, y)

    # Optimize policy:
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in agent.policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.data.item()

def preprocess_state(state):
    alpha = 1e-1
    state = 1e0 - np.exp(-alpha * state)
    return state

#-------------------------------------------------------------
# Things for plotting reward:
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# Get device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay memory:
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
CAPACITY = 50000
BATCH_SIZE = 32

# Greedy policy
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Net hyperparameters
H, actions = 300, 4

# Train parameters:
max_episodes = 5000
GAMMA = 0.99
RESET_EPISODES = 15
lr = 1e-4

# Instanciate Controller:
controller = Controller()
controller.load_circuit()

# Initialize replay memory
memory = ReplayMemory(CAPACITY)

# Initialize agent:
player = DQNPlayer(H, train=True, device=device, GUI=False)
controller.register_player(player)
player = controller.players[0]

# Optimizer:
optimizer = optim.RMSprop(player.policy.parameters(), lr=lr)

reward_list = []
best_reward = 0
for episode in range(max_episodes):
    controller.reset()
    total_reward = 0
    av_Q_val = 0
    reward_1 = 0
    done = False
    while not done:
        state = np.array(controller.get_state(player))
        state = preprocess_state(state)

        # Evaluate greedy policy:
        eps = greedy_policy(episode, EPS_START, EPS_END, EPS_DECAY)

        # Evaluate action play.
        action, Q = player.select_action(state, actions, eps)

        key = player.key_from_action(action)
        aux_count = player.count + 1
        controller.exec_action(player, key)

        if not player.crashed:
            next_state = np.array(controller.get_state(player))
            next_state = preprocess_state(next_state)

            reward = 1 * player.car.vel
            #if player.car.block != reward_1:
            #    reward = 1
            #else:
            #    reward = 0

        else:
            next_state = state.copy()
            reward = -1

        # Check if game finished:
        end = player.count >= 500 or player.laps >= 10 or player.crashed

        done = end

        # Store transition only if reward is not zero:
        reward_1 = player.car.block
        next_state = torch.tensor([next_state], device=device, dtype=torch.float)
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        state = torch.tensor([state], device=device, dtype=torch.float)
        total_reward += reward

        # Store transition in replay memory:
        store = memory.push(state, action, next_state, reward)

        # Optimize policy:
        loss = optimize(player, optimizer, memory, BATCH_SIZE, GAMMA)

        if done:
            reward_list.append(total_reward)
            #plot_durations()
            break
        ##print('Next:', next_state)
        ##print('Reward:', reward)
        ##print('end:', board.end)
        ##print()

    if total_reward > best_reward:
        #print("Checkpoint saved 'best_model.pth'. Best reward: {}".format(total_reward))
        torch.save(player.policy.state_dict(), 'best_model.pth')
        best_reward = total_reward

    #print(episode, total_reward, np.mean(av_Q_val), eps)
    if episode % RESET_EPISODES == 0:
        player.target.load_state_dict(player.policy.state_dict())

# Save model:
torch.save(player.policy.state_dict(), 'final_model.pth')
