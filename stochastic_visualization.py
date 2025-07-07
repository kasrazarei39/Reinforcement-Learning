#!/usr/bin/env python3
"""
Visualization functions for Stochastic GridWorld with windy conditions.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple
import random
import numpy as np


def visualize_wind_effects(stochastic_gw, num_simulations=100):
    """
    Visualize wind effects by showing multiple paths from the same starting point.
    
    Args:
        stochastic_gw: StochasticGridWorld instance
        num_simulations: Number of paths to simulate
    """
    print("🌪️ Visualizing Wind Effects")
    print("=" * 40)
    
    # Simulate multiple paths from start to goal
    all_paths = []
    
    for i in range(num_simulations):
        stochastic_gw.reset()
        path = [stochastic_gw.position]
        
        while True:
            row, col = stochastic_gw.position
            move = stochastic_gw.policy[row][col]
            _, state, done = stochastic_gw.action(move)
            path.append(state)
            
            if done:
                break
        
        all_paths.append(path)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw grid
    ax.set_xticks(range(stochastic_gw.N))
    ax.set_yticks(range(stochastic_gw.N))
    ax.set_xlim(-0.5, stochastic_gw.N - 0.5)
    ax.set_ylim(stochastic_gw.N - 0.5, -0.5)
    ax.grid(True, alpha=0.3)
    
    # Draw start and goal
    ax.text(stochastic_gw.start[1], stochastic_gw.start[0], 'S', 
            ha='center', va='center', fontsize=16, color='green', 
            fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgreen'))
    ax.text(stochastic_gw.terminal[1], stochastic_gw.terminal[0], 'G', 
            ha='center', va='center', fontsize=16, color='blue', 
            fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue'))
    
    # Draw all paths with different colors and transparency
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_paths)))
    
    for i, path in enumerate(all_paths):
        path_x = [pos[1] for pos in path]
        path_y = [pos[0] for pos in path]
        ax.plot(path_x, path_y, color=colors[i], alpha=0.3, linewidth=1)
    
    # Calculate and display statistics
    path_lengths = [len(path) - 1 for path in all_paths]
    avg_length = sum(path_lengths) / len(path_lengths)
    min_length = min(path_lengths)
    max_length = max(path_lengths)
    
    ax.set_title(f"Wind Effects on Agent Paths\n"
                f"Average Path Length: {avg_length:.1f} steps\n"
                f"Range: {min_length}-{max_length} steps", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def compare_deterministic_vs_stochastic_visualization():
    """
    Create side-by-side comparison of deterministic vs stochastic environments.
    """
    print("🔄 Deterministic vs Stochastic Comparison Visualization")
    print("=" * 60)
    
    # Import both environments
    from examples import create_example_gridworld
    from stochastic_examples import create_stochastic_gridworld
    
    det_gw = create_example_gridworld()
    stoch_gw = create_stochastic_gridworld()
    
    # Get paths for both environments
    def get_path(gw):
        gw.reset()
        path = [gw.position]
        while True:
            row, col = gw.position
            move = gw.policy[row][col]
            _, state, done = gw.action(move)
            path.append(state)
            if done:
                break
        return path
    
    det_path = get_path(det_gw)
    stoch_path = get_path(stoch_gw)
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Deterministic
    ax1.set_xticks(range(det_gw.N))
    ax1.set_yticks(range(det_gw.N))
    ax1.set_xlim(-0.5, det_gw.N - 0.5)
    ax1.set_ylim(det_gw.N - 0.5, -0.5)
    ax1.grid(True, alpha=0.3)
    
    # Draw start and goal for deterministic
    ax1.text(det_gw.start[1], det_gw.start[0], 'S', 
             ha='center', va='center', fontsize=16, color='green', 
             fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgreen'))
    ax1.text(det_gw.terminal[1], det_gw.terminal[0], 'G', 
             ha='center', va='center', fontsize=16, color='blue', 
             fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue'))
    
    # Draw deterministic path
    det_x = [pos[1] for pos in det_path]
    det_y = [pos[0] for pos in det_path]
    ax1.plot(det_x, det_y, 'r-', linewidth=3, markersize=8, marker='o')
    
    ax1.set_title(f"Deterministic Environment\nPath Length: {len(det_path)-1} steps", 
                  fontsize=14, fontweight='bold')
    
    # Right: Stochastic
    ax2.set_xticks(range(stoch_gw.N))
    ax2.set_yticks(range(stoch_gw.N))
    ax2.set_xlim(-0.5, stoch_gw.N - 0.5)
    ax2.set_ylim(stoch_gw.N - 0.5, -0.5)
    ax2.grid(True, alpha=0.3)
    
    # Draw start and goal for stochastic
    ax2.text(stoch_gw.start[1], stoch_gw.start[0], 'S', 
             ha='center', va='center', fontsize=16, color='green', 
             fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgreen'))
    ax2.text(stoch_gw.terminal[1], stoch_gw.terminal[0], 'G', 
             ha='center', va='center', fontsize=16, color='blue', 
             fontweight='bold', bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue'))
    
    # Draw stochastic path
    stoch_x = [pos[1] for pos in stoch_path]
    stoch_y = [pos[0] for pos in stoch_path]
    ax2.plot(stoch_x, stoch_y, 'b-', linewidth=3, markersize=8, marker='o')
    
    ax2.set_title(f"Stochastic Environment (Windy)\nPath Length: {len(stoch_path)-1} steps", 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def visualize_policy_comparison():
    """
    Visualize the difference between initial and optimal policies in stochastic environment.
    """
    print("🎯 Policy Comparison Visualization")
    print("=" * 40)
    
    from stochastic_examples import create_stochastic_gridworld
    
    # Create stochastic environment
    gw = create_stochastic_gridworld()
    
    # Get initial policy
    initial_policy = [row[:] for row in gw.policy]
    
    # Evaluate initial policy
    gw.evaluate_policy(verbose=False)
    
    # Calculate optimal policy
    gw.calculate_new_policy(verbose=False)
    optimal_policy = [row[:] for row in gw.policy]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Function to create policy heatmap
    def create_policy_heatmap(ax, policy, title):
        # Convert policy to numeric values for visualization
        policy_nums = []
        for row in policy:
            num_row = []
            for action in row:
                if action == 'N':
                    num_row.append(0)
                elif action == 'S':
                    num_row.append(1)
                elif action == 'E':
                    num_row.append(2)
                elif action == 'W':
                    num_row.append(3)
                else:
                    num_row.append(-1)
            policy_nums.append(num_row)
        
        im = ax.imshow(policy_nums, cmap='Set3', aspect='equal')
        
        # Add text annotations
        for i in range(len(policy)):
            for j in range(len(policy[0])):
                if policy[i][j]:
                    ax.text(j, i, policy[i][j], ha="center", va="center", 
                           color="black", fontweight='bold', fontsize=12)
        
        ax.set_xticks(range(len(policy[0])))
        ax.set_yticks(range(len(policy)))
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['N', 'S', 'E', 'W'])
        cbar.set_label("Actions", fontsize=12)
    
    # Left: Initial Policy
    create_policy_heatmap(ax1, initial_policy, "Initial Policy")
    
    # Right: Optimal Policy
    create_policy_heatmap(ax2, optimal_policy, "Optimal Policy (Windy)")
    
    plt.tight_layout()
    plt.show()


def show_wind_probability_model():
    """
    Visualize the wind probability model for each action.
    """
    print("🌪️ Wind Probability Model Visualization")
    print("=" * 40)
    
    from stochastic_examples import create_stochastic_gridworld
    
    gw = create_stochastic_gridworld()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    actions = ['N', 'S', 'E', 'W']
    action_names = ['North', 'South', 'East', 'West']
    
    for i, (action, name) in enumerate(zip(actions, action_names)):
        ax = axes[i]
        
        # Get outcomes and probabilities
        outcomes = gw.wind_probs[action]
        directions = []
        probabilities = []
        
        for (row_change, col_change), prob in outcomes:
            if row_change == -1 and col_change == 0:
                directions.append('N')
            elif row_change == 1 and col_change == 0:
                directions.append('S')
            elif row_change == 0 and col_change == 1:
                directions.append('E')
            elif row_change == 0 and col_change == -1:
                directions.append('W')
            elif row_change == -1 and col_change == 1:
                directions.append('NE')
            elif row_change == -1 and col_change == -1:
                directions.append('NW')
            elif row_change == 1 and col_change == 1:
                directions.append('SE')
            elif row_change == 1 and col_change == -1:
                directions.append('SW')
            elif row_change == 0 and col_change == 0:
                directions.append('Same')
            
            probabilities.append(prob)
        
        # Create bar plot
        bars = ax.bar(directions, probabilities, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{name} Action Probabilities', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, max(probabilities) * 1.2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🌪️ Stochastic GridWorld Visualization Demo")
    print("=" * 50)
    
    print("\nChoose visualization:")
    print("1. Wind Effects on Multiple Paths")
    print("2. Deterministic vs Stochastic Comparison")
    print("3. Policy Comparison (Initial vs Optimal)")
    print("4. Wind Probability Model")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        from stochastic_examples import create_stochastic_gridworld
        gw = create_stochastic_gridworld()
        visualize_wind_effects(gw)
    elif choice == "2":
        compare_deterministic_vs_stochastic_visualization()
    elif choice == "3":
        visualize_policy_comparison()
    elif choice == "4":
        show_wind_probability_model()
    else:
        print("Invalid choice. Showing wind effects...")
        from stochastic_examples import create_stochastic_gridworld
        gw = create_stochastic_gridworld()
        visualize_wind_effects(gw) 