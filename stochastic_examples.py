from typing import List, Tuple, Dict, Any
from stochastic_gridworld import StochasticGridWorld


def get_stochastic_example_config() -> Dict[str, Any]:
    """
    Get the stochastic grid world configuration with windy conditions.
    
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

    # Initial policy - can be any reasonable starting policy
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


def create_stochastic_gridworld() -> StochasticGridWorld:
    """
    Create and return a StochasticGridWorld instance with windy conditions.
    
    Returns:
        Configured StochasticGridWorld instance
    """
    config = get_stochastic_example_config()
    return StochasticGridWorld(
        N=config['size'],
        rewards=config['rewards'],
        policy=config['policy'],
        start=config['start'],
        terminal=config['terminal']
    )


def print_wind_probabilities():
    """
    Print the wind probabilities for each action to verify the model.
    """
    print("🌪️ Wind Probability Model")
    print("=" * 40)
    
    # Create a temporary instance to access wind probabilities
    config = get_stochastic_example_config()
    temp_gw = StochasticGridWorld(
        N=config['size'],
        rewards=config['rewards'],
        policy=config['policy'],
        start=config['start'],
        terminal=config['terminal']
    )
    
    for action in ['N', 'S', 'E', 'W']:
        print(f"\nAction: {action}")
        print("-" * 20)
        outcomes = temp_gw.wind_probs[action]
        total_prob = 0.0
        
        for (row_change, col_change), prob in outcomes:
            direction_name = ""
            if row_change == -1 and col_change == 0:
                direction_name = "N"
            elif row_change == 1 and col_change == 0:
                direction_name = "S"
            elif row_change == 0 and col_change == 1:
                direction_name = "E"
            elif row_change == 0 and col_change == -1:
                direction_name = "W"
            elif row_change == -1 and col_change == 1:
                direction_name = "NE"
            elif row_change == -1 and col_change == -1:
                direction_name = "NW"
            elif row_change == 1 and col_change == 1:
                direction_name = "SE"
            elif row_change == 1 and col_change == -1:
                direction_name = "SW"
            elif row_change == 0 and col_change == 0:
                direction_name = "Same"
            
            print(f"  {direction_name:4} ({row_change:2}, {col_change:2}): {prob:.3f}")
            total_prob += prob
        
        print(f"  Total probability: {total_prob:.3f}")


def test_wind_simulation():
    """
    Test the wind simulation to verify it works correctly.
    """
    print("🧪 Testing Wind Simulation")
    print("=" * 30)
    
    config = get_stochastic_example_config()
    gw = StochasticGridWorld(
        N=config['size'],
        rewards=config['rewards'],
        policy=config['policy'],
        start=config['start'],
        terminal=config['terminal']
    )
    
    # Test each action multiple times
    actions = ['N', 'S', 'E', 'W']
    test_position = (3, 3)  # Middle of grid
    
    for action in actions:
        print(f"\nTesting action: {action}")
        print("-" * 20)
        
        # Count outcomes
        outcomes_count = {}
        num_tests = 1000
        
        for _ in range(num_tests):
            new_pos = gw.move(action, test_position[0], test_position[1])
            outcomes_count[new_pos] = outcomes_count.get(new_pos, 0) + 1
        
        # Print results
        for pos, count in sorted(outcomes_count.items()):
            prob = count / num_tests
            print(f"  {pos}: {prob:.3f} ({count} times)")
        
        # Verify probabilities sum to 1
        total_prob = sum(outcomes_count.values()) / num_tests
        print(f"  Total probability: {total_prob:.3f}")


if __name__ == "__main__":
    print("🌪️ Stochastic GridWorld Examples")
    print("=" * 40)
    
    print("\n1. Wind Probability Model:")
    print_wind_probabilities()
    
    print("\n2. Wind Simulation Test:")
    test_wind_simulation() 