# GridWorld Reinforcement Learning

A modular implementation of a grid world environment for reinforcement learning, featuring policy evaluation, agent movement, and visualization capabilities.

## 🏗️ Project Structure

```
york/
├── gridworld.py      # Core GridWorld environment class
├── agent.py          # Agent behavior and policy following
├── visualization.py  # Plotting and animation functions
├── examples.py       # Example configurations and setups
├── main.py          # Main execution script
├── requirements.txt  # Project dependencies
├── README.md        # This file
└── RL.py           # Original monolithic script (legacy)
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main demonstration:**
   ```bash
   python main.py
   ```

3. **Run a simple demo:**
   ```bash
   python main.py --simple
   ```

## 📚 Module Overview

### `gridworld.py`
The core environment class that implements:
- Grid-based world with customizable size
- Movement in four directions (N, S, E, W)
- Reward system
- Policy evaluation using iterative methods
- Value function computation

**Key Methods:**
- `evaluate_policy()` - Iterative policy evaluation
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
Provides various visualization options:
- Text-based path visualization
- Animated agent movement
- Value matrix heatmaps
- Formatted value matrix printing

**Key Functions:**
- `animate_agent_path()` - Create animated visualization
- `plot_value_matrix()` - Generate heatmap of values
- `visualize_path()` - Text-based path display

### `examples.py`
Contains example configurations:
- Main 6x6 grid world example
- Simple 3x3 grid world for testing
- Factory functions for easy environment creation

**Key Functions:**
- `create_example_gridworld()` - Create main example
- `create_simple_gridworld()` - Create simple test example

### `main.py`
Main execution script with two modes:
- Full demonstration with all features
- Simple demonstration for quick testing

## 🎯 Features

- **Modular Design**: Clean separation of concerns
- **Type Hints**: Full type annotation for better code quality
- **Comprehensive Documentation**: Detailed docstrings for all functions
- **Multiple Visualization Options**: Text, heatmap, and animation
- **Performance Analysis**: Built-in metrics and analysis
- **Flexible Configuration**: Easy to create custom environments

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

### Custom Environment
```python
from gridworld import GridWorld

# Define your own configuration
rewards = [[-1, -1, 10], [-1, -2, -1], [-1, -1, -1]]
policy = [['E', 'E', 'E'], ['E', 'E', 'E'], ['E', 'E', 'E']]

# Create custom grid world
gw = GridWorld(N=3, rewards=rewards, policy=policy, 
               start=(2, 0), terminal=(0, 2))
```

## 📊 Output Examples

The system provides multiple types of output:

1. **Policy Evaluation**: Iterative convergence with delta values
2. **Value Matrix**: Numeric representation of state values
3. **Path Visualization**: Text-based grid showing agent path
4. **Performance Metrics**: Path length, total reward, efficiency
5. **Animated Visualization**: Real-time agent movement
6. **Heatmap**: Color-coded value matrix

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
4. Test with both example configurations 