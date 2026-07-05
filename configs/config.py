from memory.replay_buffer import ReplayBuffer 
from networks.q_network import NeuralNetwork


n_states , n_actions = 8 , 4
q_network = NeuralNet(n_states , n_actions)
replay_buffer = ReplayBuffer(100000)
action_dim = 4
epsilon = 0.1
batch_size = 64
gamma = 0.99
learning_rate = 5e-4
epsilon_start=1.0
epsilon_end=0.05
epsilon_decay=20000
target_update_freq = 500