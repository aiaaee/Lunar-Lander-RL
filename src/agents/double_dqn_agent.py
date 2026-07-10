from memory.replay_buffer import ReplayBuffer
from networks.q_network import NeuralNetwork
import torch 
import copy 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt


class DDQNAgent:
  def __init__(self , q_network , replay_buffer , action_dim , epsilon_decay , epsilon_start , epsilon_end , learning_rate , batch_size , gamma):
    self.q_network = q_network
    self.replay_buffer = replay_buffer
    self.action_dim = action_dim
    self.epsilon_decay = epsilon_decay
    self.epsilon_start = epsilon_start
    self.epsilon_end = epsilon_end
    self.learning_rate = learning_rate
    self.steps = 0
    self.batch_size = batch_size
    self.gamma = gamma
    self.target_network = copy.deepcopy(self.q_network )
    self.optimizer = torch.optim.Adam(self.q_network.parameters() , lr=self.learning_rate)

  def get_epsilon(self):
    return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps / self.epsilon_decay)


  def update_target_network(self):
    self.target_network.load_state_dict(
        self.q_network.state_dict()
    )

  def select_action(self , state , evaluate=False ):
    if not evaluate :
      epsilon = self.get_epsilon()
      self.steps += 1
    else :
      epsilon = 0

    if random.random() < epsilon:
      return random.randrange(self.action_dim)

    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
      q_values = self.q_network(state)
    action = torch.argmax(q_values , dim=1)
    return action.item()

  def train_step(self):
    if len(self.replay_buffer) < self.batch_size:
      return None
    states , actions , rewards , next_states , dones = self.replay_buffer.sample(self.batch_size)

    states = torch.from_numpy(states).float()
    actions = torch.from_numpy(actions).long()
    rewards = torch.from_numpy(rewards).float()
    next_states = torch.from_numpy(next_states).float()
    dones = torch.from_numpy(dones).float()

    current_q = self.q_network(states).gather(1 , actions.unsqueeze(1))
    with torch.no_grad():
      next_actions = self.q_network(next_states).argmax(dim=1 , keepdim=True)
      next_q_values = self.target_network(next_states)
      max_next_q = next_q_values.gather(1 , next_actions)
      target_q = rewards.unsqueeze(1) + (self.gamma * max_next_q * (1 - dones.unsqueeze(1)))


    loss = F.smooth_l1_loss(current_q , target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()