# GenPark Markov Decision Process Value Iteration Skill

Bellman value iteration solver computing optimal values and policies for stochastic Markov Decision Processes.

Explore more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    A[Current Value Estimates V_k] --> B[Bellman Optimality Backup]
    B --> C[max_a sum P(s'|s,a) (R + gamma * V_k(s'))]
    C --> D[Update V_k+1(s)]
    D --> E{Max Delta < Threshold?}
    E -->|No| A
    E -->|Yes: Convergence| F[Extract Optimal Greedy Policy pi*]
```

## Features
- Bellman optimality equation backup.
- Configurable discount factor gamma and convergence threshold theta.
- Pure Python standard library.
