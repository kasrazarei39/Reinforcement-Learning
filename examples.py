from typing import List, Tuple, Dict, Any
from gridworld import GridWorld


def get_example_config() -> Dict[str, Any]:
    """
    Get the example grid world configuration.
    
    Returns:
        Dictionary containing rewards, policy, start, and terminal positions
    """
    rewards = [
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -2, -1, -1],
        [-1, -1, -3, -3, -2, -1],
        [-1, -2, -4, -4, -2, -1],
        [-1, -3, -6, -6, -2, -1],
        [-1, -4, -8, -6, -4, 10]
    ]

    policy = [
        ['E', 'E', 'E', 'E', 'E', 'S'],
        ['E', 'E', 'E', 'S', 'E', 'S'],
        ['E', 'E', 'S', 'S', 'S', 'S'],
        ['E', 'S', 'S', 'S', 'S', 'S'],
        ['E', 'S', 'S', 'S', 'S', 'S'],
        ['E', 'E', 'E', 'E', 'E', 'E']
    ]

    return {
        'rewards': rewards,
        'policy': policy,
        'start': (5, 0),
        'terminal': (5, 5),
        'size': 6
    }


def create_example_gridworld() -> GridWorld:
    """
    Create and return an example GridWorld instance.
    
    Returns:
        Configured GridWorld instance
    """
    config = get_example_config()
    return GridWorld(
        N=config['size'],
        rewards=config['rewards'],
        policy=config['policy'],
        start=config['start'],
        terminal=config['terminal']
    ) 