import numpy as np
from typing import List, Tuple, Dict, Any
import random


class StochasticGridWorld:
    """
    A stochastic grid world environment for reinforcement learning with windy conditions.
    
    This class implements a grid-based environment where an agent can move
    in four directions (North, South, East, West) but actions have probabilistic outcomes
    due to wind effects.
    """
    
    def __init__(self, N: int, rewards: List[List[float]], policy: List[List[str]], 
                 start: Tuple[int, int], terminal: Tuple[int, int]):
        """
        Initialize the StochasticGridWorld environment.
        
        Args:
            N: Size of the grid (N x N)
            rewards: 2D list of rewards for each cell
            policy: 2D list of actions for each cell ('N', 'S', 'E', 'W')
            start: Starting position (row, col)
            terminal: Terminal position (row, col)
        """
        self.N = N
        self.rewards = rewards
        self.policy = policy
        self.start = start
        self.terminal = terminal
        
        # Initialize value function
        self.value = [[0.0 for _ in range(N)] for _ in range(N)]
        self.value[terminal[0]][terminal[1]] = rewards[terminal[0]][terminal[1]]
        
        # Set initial position
        self.position = start
        
        # Wind probabilities for each action
        self.wind_probs = self._initialize_wind_probabilities()
    
    def _initialize_wind_probabilities(self) -> Dict[str, List[Tuple[Tuple[int, int], float]]]:
        """
        Initialize wind probabilities for each action.
        
        Returns:
            Dictionary mapping actions to list of (outcome, probability) tuples
        """
        wind_probs = {}
        
        # North action probabilities
        north_outcomes = []
        # Primary direction: 90% success, 10% failure
        primary_success = 0.9
        primary_failure = 0.1
        
        # Lateral drift: 80% straight, 10% left, 10% right
        straight = 0.8
        left_drift = 0.1
        right_drift = 0.1
        
        # Calculate all possible outcomes and their probabilities
        # N (straight north)
        prob_n = primary_success * straight
        north_outcomes.append(((-1, 0), prob_n))
        
        # NE (north with east drift)
        prob_ne = primary_success * right_drift
        north_outcomes.append(((-1, 1), prob_ne))
        
        # NW (north with west drift)
        prob_nw = primary_success * left_drift
        north_outcomes.append(((-1, -1), prob_nw))
        
        # E (east due to failure)
        prob_e = primary_failure * right_drift
        north_outcomes.append(((0, 1), prob_e))
        
        # W (west due to failure)
        prob_w = primary_failure * left_drift
        north_outcomes.append(((0, -1), prob_w))
        
        # Same (stay in place due to failure)
        prob_same = primary_failure * straight
        north_outcomes.append(((0, 0), prob_same))
        
        wind_probs['N'] = north_outcomes
        
        # South action probabilities
        south_outcomes = []
        # S (straight south)
        prob_s = primary_success * straight
        south_outcomes.append(((1, 0), prob_s))
        
        # SE (south with east drift)
        prob_se = primary_success * right_drift
        south_outcomes.append(((1, 1), prob_se))
        
        # SW (south with west drift)
        prob_sw = primary_success * left_drift
        south_outcomes.append(((1, -1), prob_sw))
        
        # E (east due to failure)
        prob_e = primary_failure * right_drift
        south_outcomes.append(((0, 1), prob_e))
        
        # W (west due to failure)
        prob_w = primary_failure * left_drift
        south_outcomes.append(((0, -1), prob_w))
        
        # Same (stay in place due to failure)
        prob_same = primary_failure * straight
        south_outcomes.append(((0, 0), prob_same))
        
        wind_probs['S'] = south_outcomes
        
        # East action probabilities
        east_outcomes = []
        # E (straight east)
        prob_e = primary_success * straight
        east_outcomes.append(((0, 1), prob_e))
        
        # NE (east with north drift)
        prob_ne = primary_success * left_drift  # left drift = north in this context
        east_outcomes.append(((-1, 1), prob_ne))
        
        # SE (east with south drift)
        prob_se = primary_success * right_drift  # right drift = south in this context
        east_outcomes.append(((1, 1), prob_se))
        
        # N (north due to failure)
        prob_n = primary_failure * left_drift
        east_outcomes.append(((-1, 0), prob_n))
        
        # S (south due to failure)
        prob_s = primary_failure * right_drift
        east_outcomes.append(((1, 0), prob_s))
        
        # Same (stay in place due to failure)
        prob_same = primary_failure * straight
        east_outcomes.append(((0, 0), prob_same))
        
        wind_probs['E'] = east_outcomes
        
        # West action probabilities
        west_outcomes = []
        # W (straight west)
        prob_w = primary_success * straight
        west_outcomes.append(((0, -1), prob_w))
        
        # NW (west with north drift)
        prob_nw = primary_success * left_drift  # left drift = north in this context
        west_outcomes.append(((-1, -1), prob_nw))
        
        # SW (west with south drift)
        prob_sw = primary_success * right_drift  # right drift = south in this context
        west_outcomes.append(((1, -1), prob_sw))
        
        # N (north due to failure)
        prob_n = primary_failure * left_drift
        west_outcomes.append(((-1, 0), prob_n))
        
        # S (south due to failure)
        prob_s = primary_failure * right_drift
        west_outcomes.append(((1, 0), prob_s))
        
        # Same (stay in place due to failure)
        prob_same = primary_failure * straight
        west_outcomes.append(((0, 0), prob_same))
        
        wind_probs['W'] = west_outcomes
        
        return wind_probs
    
    def get_wind_outcome(self, action: str) -> Tuple[int, int]:
        """
        Simulate wind effect for a given action.
        
        Args:
            action: Action to take ('N', 'S', 'E', 'W')
            
        Returns:
            Tuple of (row_change, col_change) due to wind
        """
        outcomes = self.wind_probs[action]
        probabilities = [prob for _, prob in outcomes]
        changes = [change for change, _ in outcomes]
        
        # Choose outcome based on probabilities
        chosen_idx = np.random.choice(len(outcomes), p=probabilities)
        return changes[chosen_idx]
    
    def move(self, direction: str, row: int, col: int) -> Tuple[int, int]:
        """
        Calculate the new position after moving in a given direction with wind effects.
        
        Args:
            direction: Direction to move ('N', 'S', 'E', 'W')
            row: Current row position
            col: Current column position
            
        Returns:
            New position (row, col) after movement with wind effects
        """
        # Get wind effect
        row_change, col_change = self.get_wind_outcome(direction)
        
        # Apply changes
        new_row = row + row_change
        new_col = col + col_change
        
        # Ensure position is within grid boundaries
        new_row = max(0, min(self.N - 1, new_row))
        new_col = max(0, min(self.N - 1, new_col))
        
        return new_row, new_col
    
    def get_transition_probabilities(self, state: Tuple[int, int], action: str) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get transition probabilities for a state-action pair.
        
        Args:
            state: Current state (row, col)
            action: Action to take
            
        Returns:
            List of (next_state, probability) tuples
        """
        row, col = state
        outcomes = self.wind_probs[action]
        transitions = []
        
        for (row_change, col_change), prob in outcomes:
            new_row = row + row_change
            new_col = col + col_change
            
            # Ensure position is within grid boundaries
            new_row = max(0, min(self.N - 1, new_row))
            new_col = max(0, min(self.N - 1, new_col))
            
            next_state = (new_row, new_col)
            transitions.append((next_state, prob))
        
        return transitions
    
    def action(self, move: str) -> Tuple[float, Tuple[int, int], bool]:
        """
        Execute an action and return the result with wind effects.
        
        Args:
            move: Action to take ('N', 'S', 'E', 'W')
            
        Returns:
            Tuple of (reward, new_position, done)
        """
        row, col = self.position
        new_row, new_col = self.move(move, row, col)
        
        # Get reward for the new position
        reward = self.rewards[new_row][new_col]
        self.position = (new_row, new_col)
        
        done = self.position == self.terminal
        return reward, self.position, done
    
    def reset(self) -> None:
        """Reset the agent to the starting position."""
        self.position = self.start
    
    def evaluate_policy(self, gamma: float = 1.0, threshold: float = 1e-4, 
                       verbose: bool = True, visualize: bool = False) -> int:
        """
        Evaluate the current policy using iterative policy evaluation for stochastic environment.
        
        Args:
            gamma: Discount factor
            threshold: Convergence threshold
            verbose: Whether to print iteration details
            visualize: Whether to show value matrix visualization each iteration
            
        Returns:
            Number of iterations until convergence
        """
        # Import visualization function if needed
        if visualize:
            try:
                from visualization import plot_value_matrix_iteration
            except ImportError:
                print("Warning: visualization module not available. Setting visualize=False")
                visualize = False
        
        iteration = 0
        
        while True:
            delta = 0.0
            new_value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
            new_value[self.terminal[0]][self.terminal[1]] = self.rewards[self.terminal[0]][self.terminal[1]]

            # Update value function for all states
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        continue

                    action = self.policy[row][col]
                    
                    # Calculate expected value over all possible outcomes
                    expected_value = 0.0
                    transitions = self.get_transition_probabilities((row, col), action)
                    
                    for (next_row, next_col), prob in transitions:
                        reward = self.rewards[next_row][next_col]
                        expected_value += prob * (reward + gamma * self.value[next_row][next_col])
                    
                    new_value[row][col] = expected_value
                    delta = max(delta, abs(self.value[row][col] - expected_value))

            self.value = new_value
            iteration += 1

            if verbose:
                print(f"\nIteration {iteration} - Max Delta: {delta:.6f}")
                for r in self.value:
                    print(['{:.2f}'.format(v) for v in r])
            
            if visualize:
                plot_value_matrix_iteration(self.value, iteration, self.N)

            if delta < threshold:
                if verbose:
                    print(f"\n✅ Converged in {iteration} iterations.")
                break
        
        return iteration
    
    def calculate_new_policy(self, gamma=1.0, threshold=1e-4, verbose=True):
        """
        Calculate optimal policy using policy iteration for stochastic environment.
        """
        actions = ['N', 'S', 'E', 'W']
        new_policy = [[None for _ in range(self.N)] for _ in range(self.N)]
        iteration = 0

        while True:
            delta = 0
            new_value = [[0 for _ in range(self.N)] for _ in range(self.N)]
            new_value[self.terminal[0]][self.terminal[1]] = self.rewards[self.terminal[0]][self.terminal[1]]

            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        continue

                    best_value = float('-inf')
                    best_action = None

                    # Try all actions
                    for action in actions:
                        # Calculate expected value for this action
                        expected_value = 0.0
                        transitions = self.get_transition_probabilities((row, col), action)
                        
                        for (next_row, next_col), prob in transitions:
                            reward = self.rewards[next_row][next_col]
                            expected_value += prob * (reward + gamma * self.value[next_row][next_col])
                        
                        if expected_value > best_value:
                            best_value = expected_value
                            best_action = action

                    new_value[row][col] = best_value
                    new_policy[row][col] = best_action
                    delta = max(delta, abs(self.value[row][col] - best_value))

            self.value = new_value
            self.policy = new_policy
            iteration += 1

            if verbose:
                print(f"\nIteration {iteration} - Max Delta: {delta:.6f}")
                for r in self.value:
                    print(['{:.2f}'.format(v) for v in r])

            if delta < threshold:
                print(f"\n✅ Converged in {iteration} iterations.")
                break

        print("\nOptimal Policy:")
        for row in self.policy:
            print(row)
    
    def get_value_at(self, position: Tuple[int, int]) -> float:
        """Get the value at a specific position."""
        row, col = position
        return self.value[row][col]

    def get_policy_at(self, position: Tuple[int, int]) -> str:
        """Get the policy action at a specific position."""
        row, col = position
        return self.policy[row][col]

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if a position is valid within the grid."""
        row, col = position
        return 0 <= row < self.N and 0 <= col < self.N 