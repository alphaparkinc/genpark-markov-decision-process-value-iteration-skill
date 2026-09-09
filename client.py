"""
Markov Decision Process Value Iteration Skill Client
Pure Python Standard Library implementation of Bellman Value Iteration for Markov Decision Processes (MDP).
Computes optimal value functions V*(s) and extracts deterministic greedy policies pi*(s).
"""

from typing import List, Dict, Any, Tuple, Optional


class DiscreteMDP:
    def __init__(self, states: List[str], actions: List[str], gamma: float = 0.95):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        # Transitions: P[s][a] = [(next_s, prob, reward), ...]
        self.transitions: Dict[str, Dict[str, List[Tuple[str, float, float]]]] = {
            s: {a: [] for a in actions} for s in states
        }

    def add_transition(self, state: str, action: str, next_state: str, probability: float, reward: float):
        self.transitions[state][action].append((next_state, probability, reward))

    def value_iteration(self, theta: float = 1e-4, max_iter: int = 1000) -> Tuple[Dict[str, float], Dict[str, str], int]:
        V = {s: 0.0 for s in self.states}
        iterations = 0

        for i in range(max_iter):
            iterations += 1
            delta = 0.0
            new_V = dict(V)

            for s in self.states:
                # Terminal or dead-end states check
                q_values = []
                for a in self.actions:
                    branches = self.transitions[s][a]
                    if not branches:
                        continue
                    q_a = sum(prob * (reward + self.gamma * V[next_s]) for next_s, prob, reward in branches)
                    q_values.append(q_a)

                if q_values:
                    best_q = max(q_values)
                    delta = max(delta, abs(best_q - V[s]))
                    new_V[s] = best_q

            V = new_V
            if delta < theta:
                break

        # Extract optimal policy pi*
        policy = {}
        for s in self.states:
            best_a = None
            best_q = float("-inf")
            for a in self.actions:
                branches = self.transitions[s][a]
                if not branches:
                    continue
                q_a = sum(prob * (reward + self.gamma * V[next_s]) for next_s, prob, reward in branches)
                if q_a > best_q:
                    best_q = q_a
                    best_a = a
            policy[s] = best_a

        return V, policy, iterations
