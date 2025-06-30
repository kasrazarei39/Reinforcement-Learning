from typing import Tuple, List
from gridworld import GridWorld


def follow_policy(grid_world: GridWorld, verbose: bool = True) -> List[Tuple[int, int]]:
    """
    Follow the policy and return the path taken by the agent.
    
    Args:
        grid_world: GridWorld instance
        verbose: Whether to print step-by-step information
        
    Returns:
        List of positions visited by the agent
    """
    grid_world.reset()
    path = [grid_world.position]
    steps = 0
    
    if verbose:
        print("Following policy:")
    
    while True:
        row, col = grid_world.position
        move = grid_world.policy[row][col]
        reward, state, done = grid_world.action(move)
        steps += 1
        path.append(state)
        
        if verbose:
            print(f"Step {steps}: Move {move} → State {state}, Reward {reward}, Done {done}")
        
        if done:
            if verbose:
                print(f"\nReached terminal state in {steps} steps.")
            break
    
    return path


def get_policy_path(grid_world: GridWorld) -> List[Tuple[int, int]]:
    """
    Get the path that would be taken by following the policy without printing.
    
    Args:
        grid_world: GridWorld instance
        
    Returns:
        List of positions visited by the agent
    """
    return follow_policy(grid_world, verbose=False)


def calculate_total_reward(grid_world: GridWorld) -> float:
    """
    Calculate the total reward obtained by following the policy.
    
    Args:
        grid_world: GridWorld instance
        
    Returns:
        Total reward obtained
    """
    grid_world.reset()
    total_reward = 0
    
    while True:
        row, col = grid_world.position
        move = grid_world.policy[row][col]
        reward, state, done = grid_world.action(move)
        total_reward += reward
        
        if done:
            break
    
    return total_reward


def analyze_policy_performance(grid_world: GridWorld) -> dict:
    """
    Analyze the performance of the current policy.
    
    Args:
        grid_world: GridWorld instance
        
    Returns:
        Dictionary containing performance metrics
    """
    path = get_policy_path(grid_world)
    total_reward = calculate_total_reward(grid_world)
    
    return {
        'path_length': len(path) - 1,  # Subtract 1 because path includes start position
        'total_reward': total_reward,
        'path': path,
        'efficiency': total_reward / (len(path) - 1) if len(path) > 1 else 0
    } 