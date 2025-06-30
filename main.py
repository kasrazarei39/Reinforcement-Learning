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


def main(gw=None, policy_name="Current"):
    """Main function to run the GridWorld demonstration."""
    print(f"🚀 GridWorld Reinforcement Learning Demo - {policy_name} Policy")
    print("=" * 60)
    
    # Create the grid world environment if not provided
    if gw is None:
        print("\n📋 Creating GridWorld environment...")
        gw = create_example_gridworld()
    
    # Evaluate the policy
    print(f"\n🧮 Evaluating {policy_name.lower()} policy...")
    iterations = gw.evaluate_policy(gamma=1.0, threshold=1e-4, verbose=False, visualize=True)
    
    # Display results
    print_value_matrix(gw)
    
    # Analyze policy performance
    print(f"\n📊 Analyzing {policy_name.lower()} policy performance...")
    performance = analyze_policy_performance(gw)
    print(f"Path length: {performance['path_length']} steps")
    print(f"Total reward: {performance['total_reward']:.2f}")
    print(f"Efficiency: {performance['efficiency']:.2f} reward/step")
    
    # Follow policy step by step
    print(f"\n👟 Following {policy_name.lower()} policy step by step...")
    follow_policy(gw, verbose=True)
    
    # Visualize the path
    print(f"\n🗺️  Visualizing {policy_name.lower()} policy path...")
    visualize_path(gw)
    
    # Show both visualizations side by side
    print(f"\n📊🎬 Showing both value matrix and {policy_name.lower()} policy path animation...")
    show_both_visualizations(gw)
    
    print(f"\n✅ {policy_name} policy demo completed!")
    return gw


def run_policy_iteration_demo():
    """Run a complete policy iteration demo: initial policy → new policy."""
    print("🔄 Policy Iteration Demo")
    print("=" * 40)
    
    # Step 1: Run with initial policy
    print("\n" + "="*60)
    print("STEP 1: Running with Initial Policy")
    print("="*60)
    
    gw = main(policy_name="Initial")
    
    # Step 2: Calculate new optimal policy
    print("\n" + "="*60)
    print("STEP 2: Calculating New Optimal Policy")
    print("="*60)
    
    print("\n🧠 Running policy iteration to find optimal policy...")
    print("This will evaluate the current policy and improve it iteratively.")
    print("Press any key to continue...")
    input()
    
    gw.calculate_new_policy(gamma=1.0, threshold=1e-4, verbose=True)
    
    # Step 3: Run with new policy
    print("\n" + "="*60)
    print("STEP 3: Running with New Optimal Policy")
    print("="*60)
    
    main(gw, policy_name="Optimal")
    
    print("\n" + "="*60)
    print("🎯 Policy Iteration Demo Completed!")
    print("="*60)
    print("Summary:")
    print("- Step 1: Evaluated and analyzed the initial policy")
    print("- Step 2: Calculated the optimal policy using policy iteration")
    print("- Step 3: Evaluated and analyzed the optimal policy")
    print("="*60)


if __name__ == "__main__":
    run_policy_iteration_demo()
