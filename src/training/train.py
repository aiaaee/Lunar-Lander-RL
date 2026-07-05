from memory.replay_buffer import ReplayBuffer 
from networks.q_network import NeuralNetwork
from agents.dqn_agent import DQNAgent 
import configs.config as config

n_states , n_actions = config.n_states , config.n_actions
q_network = NeuralNet(n_states , n_actions)
replay_buffer = ReplayBuffer(100000)
action_dim = config.action_dim
epsilon = config.epsilon
batch_size = config.batch_size
gamma = config.gamma
learning_rate = config.learning_rate
epsilon_start= config.epsilon_start
epsilon_end= config.epsilon_end
epsilon_decay= config.epsilon_decay
target_update_freq = config.target_update_freq
num_episodes = 1000
global_step = 0
rewards_arr = []

Agent = DQNAgent(q_network , replay_buffer , action_dim , batch_size , learning_rate , gamma , epsilon_start , epsilon_end , epsilon_decay)

for episode in range(num_episodes):
  state , _ = env.reset()
  episode_rewards = 0
  done = False
  while not done :
    action = Agent.select_action(state)
    next_state , reward , terminated , truncated , info = env.step(action)
    done = terminated or truncated
    Agent.replay_buffer.add(state , action , reward , next_state , done )
    Agent.train_step()
    state = next_state
    episode_rewards += reward
    global_step += 1
    if global_step % target_update_freq == 0 :
      Agent.update_target_network()
  rewards_arr.append(episode_rewards)
  print(
      f"Episode : {episode + 1 }  | Reward = {episode_rewards}"
  )