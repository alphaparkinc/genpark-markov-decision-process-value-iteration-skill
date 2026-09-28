"""Markov Decision Process (MDP) Value Iteration Solver
100% Python Standard Library.
"""

class MDPValueIterationSolver:
    """Bellman optimality equation value iteration and greedy policy extractor."""
    def __init__(self, states, actions, transitions, gamma=0.95, epsilon=1e-4):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = {s: 0.0 for s in states}
        self.policy = {s: None for s in states}

    def solve(self, max_iterations=500):
        for it in range(max_iterations):
            delta = 0.0
            new_V = {}
            for s in self.states:
                best_v = float('-inf')
                for a in self.actions:
                    exp_val = 0.0
                    for prob, next_s, reward in self.transitions.get((s, a), []):
                        exp_val += prob * (reward + self.gamma * self.V[next_s])
                    if exp_val > best_v:
                        best_v = exp_val
                if best_v == float('-inf'):
                    best_v = 0.0
                new_V[s] = best_v
                delta = max(delta, abs(new_V[s] - self.V[s]))
            self.V = new_V
            if delta < self.epsilon:
                break

        for s in self.states:
            best_a = None
            best_val = float('-inf')
            for a in self.actions:
                exp_val = 0.0
                for prob, next_s, reward in self.transitions.get((s, a), []):
                    exp_val += prob * (reward + self.gamma * self.V[next_s])
                if exp_val > best_val:
                    best_val = exp_val
                    best_a = a
            self.policy[s] = best_a

        return {
            "converged_iterations": it + 1,
            "values": {s: round(v, 4) for s, v in self.V.items()},
            "policy": self.policy
        }
