#!/usr/bin/env python3
"""
Main script for Stochastic GridWorld Reinforcement Learning demonstration.

This script demonstrates policy evaluation and policy iteration in a windy
grid world environment where actions have probabilistic outcomes.
"""

from stochastic_examples import create_stochastic_gridworld
from agent import follow_policy, analyze_policy_performance
from visualization import (
    print_value_matrix, plot_value_matrix, animate_agent_path, 
    visualize_path, show_both_visualizations
)


def main(gw=None, policy_name="Current"):
    """Main function to run the Stochastic GridWorld demonstration."""
    print(f"🌪️ Stochastic GridWorld Reinforcement Learning Demo - {policy_name} Policy")
    print("=" * 70)
    
    # Create the grid world environment if not provided
    if gw is None:
        print("\n📋 Creating Stochastic GridWorld environment...")
        gw = create_stochastic_gridworld()
    
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


def run_stochastic_policy_iteration_demo():
    """Run a complete policy iteration demo for stochastic environment."""
    print("🌪️ Stochastic Policy Iteration Demo")
    print("=" * 50)
    
    print("\n🎯 This demo shows policy iteration in a WINDY grid world!")
    print("   Actions have probabilistic outcomes due to wind effects:")
    print("   - 90% chance of moving in intended direction")
    print("   - 10% chance of movement failure")
    print("   - Lateral drift: 80% straight, 10% left, 10% right")
    print("   - Creates 6 possible outcomes for each action")
    
    # Step 1: Run with initial policy
    print("\n" + "="*70)
    print("STEP 1: Running with Initial Policy (Windy Conditions)")
    print("="*70)
    
    gw = main(policy_name="Initial")
    
    # Step 2: Calculate new optimal policy
    print("\n" + "="*70)
    print("STEP 2: Calculating New Optimal Policy (Windy Conditions)")
    print("="*70)
    
    print("\n🧠 Running policy iteration to find optimal policy...")
    print("This will evaluate the current policy and improve it iteratively.")
    print("The algorithm now considers wind effects when calculating expected values.")
    print("Press any key to continue...")
    input()
    
    gw.calculate_new_policy(gamma=1.0, threshold=1e-4, verbose=True)
    
    # Step 3: Run with new policy
    print("\n" + "="*70)
    print("STEP 3: Running with New Optimal Policy (Windy Conditions)")
    print("="*70)
    
    main(gw, policy_name="Optimal")
    
    print("\n" + "="*70)
    print("🎯 Stochastic Policy Iteration Demo Completed!")
    print("="*70)
    print("Summary:")
    print("- Step 1: Evaluated and analyzed the initial policy in windy conditions")
    print("- Step 2: Calculated the optimal policy using policy iteration")
    print("- Step 3: Evaluated and analyzed the optimal policy in windy conditions")
    print("\nKey Differences from Deterministic Environment:")
    print("- Actions have probabilistic outcomes due to wind")
    print("- Policy iteration considers expected values over all possible outcomes")
    print("- Optimal policy accounts for uncertainty in the environment")
    print("="*70)


def compare_deterministic_vs_stochastic():
    """Compare deterministic vs stochastic environments."""
    print("🔄 Deterministic vs Stochastic Comparison")
    print("=" * 50)
    
    print("\nThis comparison shows the difference between deterministic and stochastic environments.")
    
    # Deterministic environment
    print("\n" + "="*50)
    print("DETERMINISTIC ENVIRONMENT")
    print("="*50)
    
    from examples import create_example_gridworld
    from gridworld import GridWorld
    
    det_gw = create_example_gridworld()
    det_gw.evaluate_policy(verbose=False)
    det_performance = analyze_policy_performance(det_gw)
    
    print(f"Deterministic - Path length: {det_performance['path_length']} steps")
    print(f"Deterministic - Total reward: {det_performance['total_reward']:.2f}")
    print(f"Deterministic - Efficiency: {det_performance['efficiency']:.2f} reward/step")
    
    # Stochastic environment
    print("\n" + "="*50)
    print("STOCHASTIC ENVIRONMENT (WINDY)")
    print("="*50)
    
    stoch_gw = create_stochastic_gridworld()
    stoch_gw.evaluate_policy(verbose=False)
    stoch_performance = analyze_policy_performance(stoch_gw)
    
    print(f"Stochastic - Path length: {stoch_performance['path_length']} steps")
    print(f"Stochastic - Total reward: {stoch_performance['total_reward']:.2f}")
    print(f"Stochastic - Efficiency: {stoch_performance['efficiency']:.2f} reward/step")
    
    # Comparison
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    
    path_diff = stoch_performance['path_length'] - det_performance['path_length']
    reward_diff = stoch_performance['total_reward'] - det_performance['total_reward']
    efficiency_diff = stoch_performance['efficiency'] - det_performance['efficiency']
    
    print(f"Path length difference: {path_diff:+d} steps")
    print(f"Total reward difference: {reward_diff:+.2f}")
    print(f"Efficiency difference: {efficiency_diff:+.2f} reward/step")
    
    print("\n💡 The stochastic environment typically shows:")
    print("   - Longer paths due to wind effects")
    print("   - Lower total rewards due to uncertainty")
    print("   - Different optimal policies that account for wind")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            compare_deterministic_vs_stochastic()
        else:
            print("Usage: python stochastic_main.py [--compare]")
            run_stochastic_policy_iteration_demo()
    else:
        # Default: Run stochastic policy iteration demo
        run_stochastic_policy_iteration_demo() 