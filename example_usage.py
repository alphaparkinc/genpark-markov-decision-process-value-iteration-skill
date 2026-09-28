from client import MDPValueIterationSolver

def main():
    states = ["A", "B", "GOAL"]
    actions = ["stay", "forward"]
    transitions = {
        ("A", "forward"): [(1.0, "B", 0.0)],
        ("A", "stay"): [(1.0, "A", 0.0)],
        ("B", "forward"): [(1.0, "GOAL", 10.0)],
        ("B", "stay"): [(1.0, "B", 0.0)],
        ("GOAL", "stay"): [(1.0, "GOAL", 0.0)]
    }
    solver = MDPValueIterationSolver(states, actions, transitions)
    res = solver.solve()
    print("MDP Value Iteration Verification:")
    print(f"Converged: {res['converged_iterations']} iterations")
    print(f"Optimal Policy: {res['policy']}")

if __name__ == "__main__":
    main()
