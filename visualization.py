import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple


def visualize_path(grid_world) -> None:
    """
    Visualize the agent's path through the grid world.
    
    Args:
        grid_world: GridWorld instance
    """
    grid_world.reset()
    visited = [[False for _ in range(grid_world.N)] for _ in range(grid_world.N)]
    visited[grid_world.position[0]][grid_world.position[1]] = True

    # Follow policy and mark visited cells
    while True:
        row, col = grid_world.position
        move = grid_world.policy[row][col]
        _, state, done = grid_world.action(move)
        visited[state[0]][state[1]] = True
        if done:
            break

    # Create path grid representation
    path_grid = [['.' for _ in range(grid_world.N)] for _ in range(grid_world.N)]
    for i in range(grid_world.N):
        for j in range(grid_world.N):
            if (i, j) == grid_world.start:
                path_grid[i][j] = 'S'
            elif (i, j) == grid_world.terminal:
                path_grid[i][j] = 'G'
            elif visited[i][j]:
                path_grid[i][j] = '*'

    print("\nGrid Path Visualization:")
    for row in path_grid:
        print(' '.join(row))


def animate_agent_path(grid_world) -> None:
    """
    Create an animated visualization of the agent's path.
    
    Args:
        grid_world: GridWorld instance
    """
    grid_world.reset()
    path = [grid_world.position]

    # Follow policy and collect path
    while True:
        row, col = grid_world.position
        move = grid_world.policy[row][col]
        _, state, done = grid_world.action(move)
        path.append(state)
        if done:
            break

    # Setup the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(range(grid_world.N))
    ax.set_yticks(range(grid_world.N))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, grid_world.N - 0.5)
    ax.set_ylim(grid_world.N - 0.5, -0.5)
    ax.grid(True, alpha=0.3)

    # Draw start and goal
    ax.text(grid_world.start[1], grid_world.start[0], 'S', 
            ha='center', va='center', fontsize=16, color='green', 
            fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgreen'))
    ax.text(grid_world.terminal[1], grid_world.terminal[0], 'G', 
            ha='center', va='center', fontsize=16, color='blue', 
            fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue'))

    # Create agent marker
    agent_marker, = ax.plot([], [], 'ro', markersize=15, markeredgecolor='darkred', 
                           markeredgewidth=2)

    def update(frame):
        """Update function for animation."""
        row, col = path[frame]
        agent_marker.set_data(col, row)
        return agent_marker,

    # Create and show animation
    ani = animation.FuncAnimation(fig, update, frames=len(path), 
                                 interval=800, repeat=False, blit=True)
    plt.title("Agent Path Animation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_value_matrix(grid_world, block: bool = True) -> None:
    """
    Create a heatmap visualization of the value matrix.
    
    Args:
        grid_world: GridWorld instance
        block: Whether to block execution while showing the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(grid_world.value, cmap='RdYlBu_r', aspect='equal')
    
    # Add text annotations
    for i in range(grid_world.N):
        for j in range(grid_world.N):
            text = ax.text(j, i, f'{grid_world.value[i][j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Customize plot
    ax.set_xticks(range(grid_world.N))
    ax.set_yticks(range(grid_world.N))
    ax.set_title("Value Matrix Heatmap", fontsize=14, fontweight='bold')
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value", fontsize=12)
    
    plt.tight_layout()
    plt.show(block=block)


def print_value_matrix(grid_world) -> None:
    """
    Print the value matrix in a formatted way.
    
    Args:
        grid_world: GridWorld instance
    """
    print("\nValue Matrix:")
    for row in grid_world.value:
        print(['{:.2f}'.format(v) for v in row])


def show_both_visualizations(grid_world) -> None:
    """
    Show both value matrix heatmap and agent path animation side by side.
    
    Args:
        grid_world: GridWorld instance
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: Value matrix heatmap
    im1 = ax1.imshow(grid_world.value, cmap='RdYlBu_r', aspect='equal')
    
    # Add text annotations to heatmap
    for i in range(grid_world.N):
        for j in range(grid_world.N):
            ax1.text(j, i, f'{grid_world.value[i][j]:.1f}',
                    ha="center", va="center", color="black", fontweight='bold')
    
    ax1.set_xticks(range(grid_world.N))
    ax1.set_yticks(range(grid_world.N))
    ax1.set_title("Value Matrix Heatmap", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Column", fontsize=12)
    ax1.set_ylabel("Row", fontsize=12)
    
    # Add colorbar to heatmap
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Value", fontsize=12)
    
    # Right subplot: Agent path animation
    grid_world.reset()
    path = [grid_world.position]
    
    # Follow policy and collect path
    while True:
        row, col = grid_world.position
        move = grid_world.policy[row][col]
        _, state, done = grid_world.action(move)
        path.append(state)
        if done:
            break
    
    # Setup animation subplot
    ax2.set_xticks(range(grid_world.N))
    ax2.set_yticks(range(grid_world.N))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xlim(-0.5, grid_world.N - 0.5)
    ax2.set_ylim(grid_world.N - 0.5, -0.5)
    ax2.grid(True, alpha=0.3)
    
    # Draw start and goal
    ax2.text(grid_world.start[1], grid_world.start[0], 'S', 
             ha='center', va='center', fontsize=16, color='green', 
             fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgreen'))
    ax2.text(grid_world.terminal[1], grid_world.terminal[0], 'G', 
             ha='center', va='center', fontsize=16, color='blue', 
             fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue'))
    
    # Create agent marker
    agent_marker, = ax2.plot([], [], 'ro', markersize=15, markeredgecolor='darkred', 
                            markeredgewidth=2)
    
    def update(frame):
        """Update function for animation."""
        row, col = path[frame]
        agent_marker.set_data(col, row)
        return agent_marker,
    
    ax2.set_title("Agent Path Animation", fontsize=14, fontweight='bold')
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(path), 
                                 interval=800, repeat=False, blit=True)
    
    plt.tight_layout()
    plt.show()


def show_visualizations_sequence(grid_world, delay: float = 3.0) -> None:
    """
    Show visualizations in sequence with a delay between them.
    
    Args:
        grid_world: GridWorld instance
        delay: Delay in seconds between visualizations
    """
    import time
    
    # Show value matrix first
    print("\n📈 Showing value matrix heatmap...")
    plot_value_matrix(grid_world, block=False)
    time.sleep(delay)
    
    # Close the value matrix plot
    plt.close()
    
    # Show agent path animation
    print("🎬 Showing agent path animation...")
    animate_agent_path(grid_world) 