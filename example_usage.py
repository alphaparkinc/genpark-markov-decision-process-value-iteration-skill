"""
Demonstration of Markov Decision Process Value Iteration Skill
"""

from client import DiscreteMDP

def main():
    print("=== Solving Gridworld MDP with Bellman Value Iteration ===")
    states = ["S0", "S1", "GOAL", "PIT"]
    actions = ["LEFT", "RIGHT"]
    mdp = DiscreteMDP(states, actions, gamma=0.9)

    # Transitions from S0
    mdp.add_transition("S0", "RIGHT", "S1", 1.0, -0.04)
    mdp.add_transition("S0", "LEFT", "S0", 1.0, -0.04)

    # Transitions from S1
    mdp.add_transition("S1", "RIGHT", "GOAL", 0.8, 1.0)
    mdp.add_transition("S1", "RIGHT", "PIT", 0.2, -1.0)
    mdp.add_transition("S1", "LEFT", "S0", 1.0, -0.04)

    V_star, policy_star, iters = mdp.value_iteration()
    print(f"Convergence achieved in {iters} iterations.")
    print("Optimal State Values V*(s):")
    for s, v in sorted(V_star.items()):
        print(f"  {s}: {v:.4f}")

    print("\nOptimal Policy pi*(s):")
    for s, a in sorted(policy_star.items()):
        print(f"  {s} -> {a}")

    assert policy_star["S0"] == "RIGHT"
    print("\nMarkov Decision Process Value Iteration Verification PASS!")

if __name__ == "__main__":
    main()
