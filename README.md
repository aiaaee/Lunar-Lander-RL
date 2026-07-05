# Lunar Lander with DQN and Double DQN

## Project Overview 

This project implements and evaluates **Deep Q-Network (DQN)** and **Double Deep Q-Network (Double DQN)** to solve the **Lunar Lander** environment. The objective of the task is to train an autonomous agent to perform a safe and stable landing by maximizing the cumulative reward through interaction with the environment. The project demonstrates how value-based reinforcement learning algorithms can learn an effective landing policy from trial-and-error experience.

The implementation is developed entirely in **Python** using **PyTorch** for the neural network components, while **NumPy** and native Python are used to build the reinforcement learning pipeline, including the agent, replay buffer, and training process. The DQN implementation incorporates several key techniques such as **Experience Replay**, **Target Networks**, and an **ε-greedy exploration strategy** to improve training stability and learning efficiency.

The primary goal of this project is to compare the performance of DQN and Double DQN under the same environment and training conditions. In addition to the performance comparison, the project aims to provide a clear understanding of agent behavior during learning and to illustrate the transition from classical tabular reinforcement learning concepts to deep reinforcement learning methods through a clean, modular implementation.

<img width="600" height="400" alt="lunar_lander" src="https://github.com/user-attachments/assets/3eecd699-c893-41fe-b7c9-a03e578af796" />

## Demo 

The following demonstrations show the trained agents interacting with the Lunar Lander environment after completing the training process.


## Features 

- Implementation of Deep Q-Network (DQN) from scratch using PyTorch  
- Implementation of Double DQN to reduce overestimation bias  
- Experience Replay for stable training  
- Target Network for improved convergence  
- Epsilon-Greedy policy for exploration vs exploitation  
- Modular and clean project structure  
- Training and evaluation pipelines separated  
- Performance comparison between DQN and Double DQN  


## Project Structure 

Lunar-Lander-RL/
│
├── configs/ # Hyperparameters for DQN and Double DQN
│ ├── dqn.yaml
│ └── double_dqn.yaml
│
├── src/
│ ├── agents/ # DQN and Double DQN agents
│ │ ├── dqn_agent.py
│ │ └── double_dqn_agent.py
│ │
│ ├── networks/ # Neural network architecture
│ │ └── q_network.py
│ │
│ ├── memory/ # Experience Replay buffer
│ │ └── replay_buffer.py
│ │
│ ├── training/ # Training loops
│ │ ├── train_dqn.py
│ │ └── train_double_dqn.py
│ │
│ ├── evaluation/ # Evaluation scripts
│ │ └── evaluate.py
│ │
│ └── utils/ # Helper functions (logging, plotting, etc.)
│ ├── seed.py
│ ├── plotting.py
│ └── logger.py
│
├── models/ # Saved trained models (.pt files)
│ ├── dqn/
│ └── double_dqn/
│
├── results/ # Training results (plots, logs, videos)
│ ├── plots/
│ └── videos/
│
├── notebooks/ # Experiments and testing
│
├── README.md
└── requirements.txt

--- 
```
The project is structured in a modlar way to clearly separate agent logic, neural networks, training pipeline, and evaluation process. 
```
