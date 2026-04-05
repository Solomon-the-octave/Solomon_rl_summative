Solomon RL Summative Project

This project explores how reinforcement learning can be used to simulate decision-making in urban planning. The goal is to model how a city can balance multiple competing priorities such as housing, transport, infrastructure, environmental sustainability, and budget constraints.

Project Overview

Urban systems are highly interconnected, meaning that improving one aspect often creates trade-offs in others. In this project, I designed a custom reinforcement learning environment where an agent acts as a city decision-maker. The agent selects different interventions (e.g., improving transport or housing) and learns over time how to make better decisions based on rewards and penalties.

The objective is not to maximize a single variable, but to maintain a balance across multiple indicators while avoiding negative outcomes like pollution or budget depletion.

Environment Design
Agent

The agent represents a city planner making sequential decisions. It learns from interactions with the environment and adjusts its strategy over time.

Action Space (Discrete)
Build housing
Improve transport
Upgrade informal settlements
Expand water and sanitation
Do nothing

Each action impacts multiple aspects of the system, introducing real-world trade-offs.

Observation Space

The state is represented as a numerical vector including:

Population pressure
Housing level
Transport efficiency
Water and sanitation
Green space
Informal settlements
Pollution
Budget
Reward Function

The reward is designed to encourage balanced development:

Positive rewards:

Improved housing and infrastructure
Reduction in informal settlements

Negative rewards:

Increased pollution
Budget depletion
Worsening population pressure
Models Implemented

The project compares both value-based and policy-based methods:

1. Deep Q-Network (DQN)
Uses neural networks to estimate Q-values
Experience replay for stability
Epsilon-greedy exploration
2. REINFORCE (Policy Gradient)
Learns directly from episode rewards
High variance and less stable
3. PPO (Proximal Policy Optimization)
Uses clipped objective function
More stable than REINFORCE
4. A2C (Actor-Critic)
Combines value and policy learning
Moderate performance and stability
Key Results
DQN achieved the highest mean reward and most stable performance
PPO showed consistent but moderate performance
A2C performed reasonably but did not outperform DQN
REINFORCE had the highest variability and unstable learning

Overall, DQN provided the best balance between stability and performance in this environment.

Visualizations

The project includes:

Learning rate vs mean reward plots (DQN & PPO)
Training stability plots (REINFORCE)
Cumulative reward comparison across models

These help illustrate how each model behaves during training and how stable their learning process is.

How to Run

Install dependencies:
pip install -r requirements.txt
Run the simulation:
python main.py

Project Structure

project_root/
│
├── environment/
│   ├── custom_env.py
│   └── rendering.py
│
├── training/
│   ├── dqn_training.py
│   └── pg_training.py
│
├── models/
│   ├── dqn/
│   └── pg/
│
├── main.py
├── requirements.txt
└── README.md

Final Thoughts

This project highlights how reinforcement learning can be applied to complex, real-world systems where decisions involve trade-offs rather than simple optimization. It also shows the importance of stability and proper tuning when comparing different RL approaches.
