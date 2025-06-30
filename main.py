#!/usr/bin/env python3
"""
Main script for GridWorld Reinforcement Learning demonstration.

This script demonstrates policy evaluation, agent movement, and visualization
using a modular grid world environment.
"""

from examples import create_example_gridworld
from agent import follow_policy, analyze_policy_performance
from visualization import (
    print_value_matrix, plot_value_matrix, animate_agent_path, 
    visualize_path, show_both_visualizations, show_visualizations_sequence
)


def main():
    """Main function to run the GridWorld demonstration."""
    print("🚀 GridWorld Reinforcement Learning Demo")
    print("=" * 50)
    
    # Create the grid world environment
    print("\n📋 Creating GridWorld environment...")
    gw = create_example_gridworld()
    
    # Evaluate the policy
    print("\n🧮 Evaluating policy...")
    iterations = gw.evaluate_policy(gamma=1.0, threshold=1e-4, verbose=False, visualize=True)
    
    # Display results
    print_value_matrix(gw)
    
    # Analyze policy performance
    print("\n📊 Analyzing policy performance...")
    performance = analyze_policy_performance(gw)
    print(f"Path length: {performance['path_length']} steps")
    print(f"Total reward: {performance['total_reward']:.2f}")
    print(f"Efficiency: {performance['efficiency']:.2f} reward/step")
    
    # Follow policy step by step
    print("\n👟 Following policy step by step...")
    follow_policy(gw, verbose=True)
    
    # Visualize the path
    print("\n🗺️  Visualizing agent path...")
    visualize_path(gw)
    
    # Show both visualizations side by side
    print("\n📊🎬 Showing both value matrix and agent path animation...")
    show_both_visualizations(gw)
    
    print("\n✅ Demo completed!")


def run_visualization_demo():
    """Run a demo focused on different visualization options."""
    print("🎨 Visualization Options Demo")
    print("=" * 35)
    
    from examples import create_example_gridworld
    
    # Create grid world
    gw = create_example_gridworld()
    gw.evaluate_policy(verbose=False)
    
    print("\nChoose visualization option:")
    print("1. Value matrix heatmap only")
    print("2. Agent path animation only")
    print("3. Both visualizations side by side")
    print("4. Visualizations in sequence")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        plot_value_matrix(gw)
    elif choice == "2":
        animate_agent_path(gw)
    elif choice == "3":
        show_both_visualizations(gw)
    elif choice == "4":
        show_visualizations_sequence(gw)
    else:
        print("Invalid choice. Showing both side by side.")
        show_both_visualizations(gw)


def run_iterative_visualization_demo():
    """Run a demo showing value matrix visualization during policy evaluation."""
    print("🎬 Iterative Value Matrix Visualization Demo")
    print("=" * 50)
    
    from examples import create_example_gridworld
    
    print("\n📊 Running policy evaluation with 6x6 grid...")
    gw = create_example_gridworld()
    
    print("\n🎬 Watch the value matrix evolve during policy evaluation!")
    print("Each iteration will show a new heatmap visualization.")
    print("Press any key to continue...")
    input()
    
    # Evaluate policy with visualization enabled
    iterations = gw.evaluate_policy(gamma=1.0, threshold=1e-4, 
                                   verbose=True, visualize=True)
    
    print(f"\n✅ Policy evaluation completed in {iterations} iterations!")
    print("\n🎯 Final value matrix:")
    print_value_matrix(gw)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--visualization":
            run_visualization_demo()
        elif sys.argv[1] == "--iterative":
            run_iterative_visualization_demo()
        else:
            print("Usage: python main.py [--visualization|--iterative]")
            main()
    else:
        main() 