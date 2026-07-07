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
- Implementation of Double DQN (DDQN) to reduce overestimation bias  
- Experience Replay for stable training  
- Target Network for improved convergence  
- Epsilon-Greedy policy for exploration vs exploitation  
- Modular and clean project structure  
- Training and evaluation pipelines separated  
- Performance comparison between DQN and Double DQN  


## Project Structure

```text
Lunar-Lander-RL/
├── configs/
│   ├── dqn.yaml
│   └── double_dqn.yaml
│
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   └── double_dqn_agent.py
│   │
│   ├── networks/
│   │   └── q_network.py
│   │
│   ├── memory/
│   │   └── replay_buffer.py
│   │
│   ├── training/
│   │   ├── train_dqn.py
│   │   └── train_double_dqn.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── plotting.py
│       ├── logger.py
│       └── seed.py
│
├── models/
│   ├── dqn/
│   └── double_dqn/
│
├── results/
│   ├── plots/
│   ├── videos/
│   └── logs/
│
├── README.md
├── requirements.txt
└── LICENSE
```

The project follows a modular structure that separates the agent, neural network, replay buffer, training pipeline, evaluation, and utility functions, making the codebase easier to maintain and extend.


## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Lunar-Lander-RL.git
cd Lunar-Lander-RL
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the agent

To train the DQN agent:

```bash
python src/training/train_dqn.py
```

To train the Double DQN agent:

```bash
python src/training/train_double_dqn.py
```

### 4. Evaluate a trained model

```bash
python src/evaluation/evaluate.py
```

## Future Work

This project provides a baseline implementation of DQN and Double DQN for the Lunar Lander environment. Future improvements may include:

* Implementing **Dueling DQN** to improve value estimation.
* Integrating **Prioritized Experience Replay (PER)** for more efficient sampling.
* Extending the project with advanced reinforcement learning algorithms such as **Rainbow DQN** and **PPO**.
* Performing comprehensive hyperparameter tuning and performance evaluation.
* Comparing the implemented algorithms using quantitative metrics and training curves.
* Improving the visualization of the agent's behavior with additional demonstrations and analysis.


