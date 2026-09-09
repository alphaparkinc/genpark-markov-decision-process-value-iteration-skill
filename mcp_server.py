"""
MCP Server for Markov Decision Process Value Iteration Skill
"""

import json
import sys
from client import DiscreteMDP

def handle_call(name: str, args: dict) -> dict:
    if name == "solve_mdp":
        st = args.get("states", ["S0", "S1"])
        act = args.get("actions", ["A0", "A1"])
        gamma = args.get("gamma", 0.9)
        mdp = DiscreteMDP(st, act, gamma)
        for t in args.get("transitions", []):
            mdp.add_transition(t[0], t[1], t[2], t[3], t[4])
        v, p, iters = mdp.value_iteration()
        return {"values": v, "policy": p, "iterations": iters}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
