# GridWorld Reinforcement Learning

A modular implementation of a grid world environment for reinforcement learning, featuring policy evaluation, policy iteration, and comprehensive visualization capabilities.

## 🏗️ Project Structure

```
Reinforcement-Learning/
├── gridworld.py      # Core GridWorld environment class
├── agent.py          # Agent behavior and policy analysis
├── visualization.py  # Plotting and animation functions
├── examples.py       # Example configurations and setups
├── main.py          # Main execution script
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the policy iteration demonstration:**
   ```bash
   python main.py
   ```

## 📚 Module Overview

### `gridworld.py`
The core environment class that implements:
- Grid-based world with customizable size (6×6 default)
- Movement in four directions (N, S, E, W)
- Reward system with negative costs and positive goal reward
- Policy evaluation using iterative methods
- Policy iteration for finding optimal policies
- Value function computation

**Key Methods:**
- `evaluate_policy()` - Iterative policy evaluation with visualization support
- `calculate_new_policy()` - Policy iteration (evaluation + improvement)
- `action()` - Execute an action and get reward
- `move()` - Calculate new position after movement
- `reset()` - Reset agent to starting position

### `agent.py`
Handles agent behavior and policy analysis:
- Policy following with step-by-step tracking
- Performance analysis (path length, total reward, efficiency)
- Path extraction for visualization

**Key Functions:**
- `follow_policy()` - Follow policy with optional verbose output
- `analyze_policy_performance()` - Calculate performance metrics
- `calculate_total_reward()` - Compute total reward for a policy

### `visualization.py`
Provides comprehensive visualization options:
- Text-based path visualization
- Animated agent movement
- Value matrix heatmaps
- Iterative value matrix visualization during policy evaluation
- Side-by-side visualizations

**Key Functions:**
- `plot_value_matrix_iteration()` - Show value matrix evolution
- `animate_agent_path()` - Create animated visualization
- `plot_value_matrix()` - Generate heatmap of values
- `visualize_path()` - Text-based path display
- `show_both_visualizations()` - Combined heatmap and animation

### `examples.py`
Contains the main grid world configuration:
- 6×6 grid world with challenging reward structure
- Factory function for easy environment creation

**Key Functions:**
- `create_example_gridworld()` - Create main example environment

### `main.py`
Main execution script featuring:
- Complete policy iteration demonstration
- Three-step workflow: initial policy → policy improvement → optimal policy
- Comprehensive analysis and visualization

## 🎯 Features

- **Policy Iteration Demo**: Complete workflow from initial to optimal policy
- **Interactive Visualization**: Watch value matrices evolve during evaluation
- **Performance Analysis**: Compare policy performance before and after improvement
- **Modular Design**: Clean separation of concerns
- **Type Hints**: Full type annotation for better code quality
- **Comprehensive Documentation**: Detailed docstrings for all functions
- **Multiple Visualization Options**: Text, heatmap, and animation
- **Real-time Learning**: See algorithms converge step-by-step

## 🔧 Usage Examples

### Basic Usage
```python
from examples import create_example_gridworld
from agent import follow_policy
from visualization import print_value_matrix

# Create environment
gw = create_example_gridworld()

# Evaluate policy
gw.evaluate_policy()

# Display results
print_value_matrix(gw)

# Follow policy
follow_policy(gw)
```

### Policy Iteration
```python
from examples import create_example_gridworld

# Create environment
gw = create_example_gridworld()

# Evaluate initial policy
gw.evaluate_policy(visualize=True)

# Find optimal policy
gw.calculate_new_policy(verbose=True)

# Evaluate optimal policy
gw.evaluate_policy(visualize=True)
```

### Custom Environment
```python
from gridworld import GridWorld

# Define your own configuration
rewards = [[-1, -1, -1, 10], [-1, -2, -1, -1], [-1, -1, -2, -1], [-1, -1, -1, -1]]
policy = [['E', 'E', 'E', 'E'], ['E', 'E', 'E', 'E'], ['E', 'E', 'E', 'E'], ['E', 'E', 'E', 'E']]

# Create custom grid world
gw = GridWorld(N=4, rewards=rewards, policy=policy, 
               start=(3, 0), terminal=(0, 3))
```

## 📊 Output Examples

The system provides multiple types of output:

1. **Policy Evaluation**: Iterative convergence with delta values
2. **Value Matrix**: Numeric representation of state values
3. **Path Visualization**: Text-based grid showing agent path
4. **Performance Metrics**: Path length, total reward, efficiency
5. **Animated Visualization**: Real-time agent movement
6. **Heatmap**: Color-coded value matrix
7. **Iterative Visualization**: Value matrix evolution during evaluation
8. **Policy Comparison**: Before/after policy improvement analysis

## 🔄 Policy Iteration Workflow

The main demonstration follows a three-step process:

### Step 1: Initial Policy Analysis
- Create GridWorld environment with default policy
- Evaluate policy using iterative policy evaluation
- Display value matrix and performance metrics
- Show agent path visualization and animation

### Step 2: Policy Improvement
- Run policy iteration algorithm
- Find optimal policy through evaluation and improvement cycles
- Show convergence process with iteration details

### Step 3: Optimal Policy Analysis
- Evaluate the new optimal policy
- Compare performance metrics with initial policy
- Visualize improved agent behavior
- Demonstrate efficiency gains

## 🛠️ Development

The code is designed to be easily extensible:

- Add new algorithms by extending the `GridWorld` class
- Create new visualization types in `visualization.py`
- Add more example configurations in `examples.py`
- Implement new agent behaviors in `agent.py`

## 📝 Requirements

- Python 3.7+
- matplotlib (for visualization)
- numpy (for numerical operations)

## 🤝 Contributing

The modular structure makes it easy to contribute:
1. Add new features to appropriate modules
2. Maintain type hints and documentation
3. Follow the existing code style
4. Test with the example configuration

## 🎓 Educational Value

This project demonstrates key reinforcement learning concepts:

- **Markov Decision Processes (MDPs)**: States, actions, rewards, transitions
- **Policy Evaluation**: Computing value functions for given policies
- **Policy Iteration**: Finding optimal policies through evaluation and improvement
- **Bellman Equations**: Value function updates
- **Convergence Properties**: Algorithm termination conditions
- **Performance Analysis**: Comparing different policies

Perfect for understanding fundamental RL algorithms and their practical implementation! 