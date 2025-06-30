from typing import List, Tuple, Optional


class GridWorld:
    """
    A grid world environment for reinforcement learning.
    
    This class implements a grid-based environment where an agent can move
    in four directions (North, South, East, West) and receives rewards
    based on the cells it visits.
    """
    
    def __init__(self, N: int, rewards: List[List[float]], policy: List[List[str]], 
                 start: Tuple[int, int], terminal: Tuple[int, int]):
        """
        Initialize the GridWorld environment.
        
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

    def move(self, direction: str, row: int, col: int) -> Tuple[int, int]:
        """
        Calculate the new position after moving in a given direction.
        
        Args:
            direction: Direction to move ('N', 'S', 'E', 'W')
            row: Current row position
            col: Current column position
            
        Returns:
            New position (row, col) after movement
        """
        if direction == 'N' and row > 0:
            row -= 1
        elif direction == 'S' and row < self.N - 1:
            row += 1
        elif direction == 'W' and col > 0:
            col -= 1
        elif direction == 'E' and col < self.N - 1:
            col += 1
        return row, col

    def action(self, move: str) -> Tuple[float, Tuple[int, int], bool]:
        """
        Execute an action and return the result.
        
        Args:
            move: Action to take ('N', 'S', 'E', 'W')
            
        Returns:
            Tuple of (reward, new_position, done)
        """
        row, col = self.position
        new_row, new_col = self.move(move, row, col)

        # Determine reward based on whether movement occurred
        if (new_row, new_col) == (row, col):
            reward = self.rewards[row][col]  # Hit boundary, stay in place
        else:
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
        Evaluate the current policy using iterative policy evaluation.
        
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

                    move = self.policy[row][col]
                    next_row, next_col = self.move(move, row, col)
                    reward = self.rewards[next_row][next_col]
                    value = reward + gamma * self.value[next_row][next_col]
                    new_value[row][col] = value
                    delta = max(delta, abs(self.value[row][col] - value))

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

    def get_value_at(self, position: Tuple[int, int]) -> float:
        """
        Get the value at a specific position.
        
        Args:
            position: (row, col) position
            
        Returns:
            Value at the specified position
        """
        row, col = position
        return self.value[row][col]

    def get_policy_at(self, position: Tuple[int, int]) -> str:
        """
        Get the policy action at a specific position.
        
        Args:
            position: (row, col) position
            
        Returns:
            Policy action at the specified position
        """
        row, col = position
        return self.policy[row][col]

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is valid within the grid.
        
        Args:
            position: (row, col) position to check
            
        Returns:
            True if position is valid, False otherwise
        """
        row, col = position
        return 0 <= row < self.N and 0 <= col < self.N 