from memory.replay_buffer import ReplayBuffer
from networks.q_network import NeuralNetwork
import torch 
import copy 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 

class DQNAgent:
  def __init__(self , q_network , replay_buffer , action_dim , batch_size , learning_rate , gamma , start_epsilon , end_epsilon , epsilon_decay):
    self.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
    self.q_network = q_network.to(self.device)
    self.replay_buffer = replay_buffer
    self.action_dim = action_dim
    self.steps = 0
    self.start_epsilon = start_epsilon
    self.end_epsilon = end_epsilon
    self.epsilon_decay = epsilon
    self.batch_size = batch_size
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.target_network = copy.deepcopy(self.q_network).to(self.device)
    self.optimizer = torch.optim.Adam(params=self.q_network.parameters() , lr=self.learning_rate)

    for param in self.target_network.parameters():
      param.require_grad = False

  def update_target_network(self):
    self.target_network.load_state_dict(self.q_network.state_dict())

  def get_epsilon(self):
    epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-self.steps / self.epsilon_decay)
    return epsilon

  def select_action(self , state , evaluate=False):
    if not evaluate :
      epsilon = self.get_epsilon()
      self.steps += 1
    else :
      epsilon = 0

    if random.random() < epsilon :
      return random.randrange(self.action_dim)
    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    with torch.no_grad():
      q_values = self.q_network(state)
    action = torch.argmax(q_values , dim=1)
    return action.item()

  def train_step(self):
    if len(self.replay_buffer) < self.batch_size :
      return None

    states , actions , rewards , next_states , dones = self.replay_buffer.sample(self.batch_size)

    states = torch.from_numpy(states).float().to(self.device)
    actions = torch.from_numpy(actions).long().to(self.device)
    rewards = torch.from_numpy(rewards).float().to(self.device)
    next_states = torch.from_numpy(next_states).float().to(self.device)
    dones = torch.from_numpy(dones).float().to(self.device)

    current_q = self.q_network(states).gather(dim=1 , index=actions.unsqueeze(1))

    with torch.no_grad():
      max_next_q = self.target_network(next_states).max(dim=1 , keepdim=True)[0]
      target_q = rewards.unsqueeze(1) + (self.gamma * max_next_q * (1 - dones.unsqueeze(1)))

    loss = F.smooth_l1_loss(current_q , target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()
  def __init__(self , q_network , replay_buffer , action_dim , batch_size , learning_rate , gamma , start_epsilon , end_epsilon , epsilon_decay):
    self.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
    self.q_network = q_network.to(self.device)
    self.replay_buffer = replay_buffer
    self.action_dim = action_dim
    self.steps = 0
    self.start_epsilon = start_epsilon
    self.end_epsilon = end_epsilon
    self.epsilon_decay = epsilon
    self.batch_size = batch_size
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.target_network = copy.deepcopy(self.q_network).to(self.device)
    self.optimizer = torch.optim.Adam(params=self.q_network.parameters() , lr=self.learning_rate)

    for param in self.target_network.parameters():
      param.require_grad = False

  def update_target_network(self):
    self.target_network.load_state_dict(self.q_network.state_dict())

  def get_epsilon(self):
    epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-self.steps / self.epsilon_decay)
    return epsilon

  def select_action(self , state):
    epsilon = self.get_epsilon()
    self.steps += 1
    if random.random() < epsilon :
      return random.randrange(self.action_dim)
    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    with torch.no_grad():
      q_values = self.q_network(state)
    action = torch.argmax(q_values , dim=1)
    return action.item()

  def train_step(self):
    if len(self.replay_buffer) < self.batch_size :
      return None

    states , actions , rewards , next_states , dones = self.replay_buffer.sample(self.batch_size)

    states = torch.from_numpy(states).float().to(self.device)
    actions = torch.from_numpy(actions).long().to(self.device)
    rewards = torch.from_numpy(rewards).float().to(self.device)
    next_states = torch.from_numpy(next_states).float().to(self.device)
    dones = torch.from_numpy(dones).float().to(self.device)

    current_q = self.q_network(states).gather(dim=1 , index=actions.unsqueeze(1))

    with torch.no_grad():
      max_next_q = self.target_network(next_states).max(dim=1 , keepdim=True)[0]
      target_q = rewards.unsqueeze(1) + (self.gamma * max_next_q * (1 - dones.unsqueeze(1)))

    loss = F.smooth_l1_loss(current_q , target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()