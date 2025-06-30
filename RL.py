#!/usr/bin/env python3
"""
Legacy wrapper for GridWorld RL demonstration.

This file maintains backward compatibility with the original monolithic script
while using the new modular structure under the hood.
"""

import warnings

# Import from new modular structure
from examples import create_example_gridworld
from agent import follow_policy
from visualization import print_value_matrix, visualize_path, animate_agent_path


def main():
    """Main function that replicates the original RL.py behavior."""
    warnings.warn(
        "This is the legacy RL.py script. Consider using main.py for the new modular structure.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create the grid world environment
    gw = create_example_gridworld()
    
    # Evaluate the policy (replicates original behavior)
    gw.evaluate_policy()
    
    # Display value matrix
    print_value_matrix(gw)
    
    # Uncomment to visualize path or step-by-step moves
    # follow_policy(gw)
    # visualize_path(gw)
    animate_agent_path(gw)


if __name__ == "__main__":
    main()
