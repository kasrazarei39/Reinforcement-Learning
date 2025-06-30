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
    iterations = gw.evaluate_policy(gamma=1.0, threshold=1e-4, verbose=True)
    
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


def run_simple_demo():
    """Run a simple demonstration with minimal output."""
    print("🚀 Simple GridWorld Demo")
    print("=" * 30)
    
    from examples import create_simple_gridworld
    
    # Create simple 3x3 grid world
    gw = create_simple_gridworld()
    
    # Evaluate policy
    gw.evaluate_policy(verbose=False)
    
    # Show results
    print_value_matrix(gw)
    visualize_path(gw)
    
    # Follow policy
    follow_policy(gw, verbose=True)
    
    # Show visualizations in sequence
    print("\n📊🎬 Showing visualizations in sequence...")
    show_visualizations_sequence(gw, delay=2.0)


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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--simple":
            run_simple_demo()
        elif sys.argv[1] == "--visualization":
            run_visualization_demo()
        else:
            print("Usage: python main.py [--simple|--visualization]")
            main()
    else:
        main() 