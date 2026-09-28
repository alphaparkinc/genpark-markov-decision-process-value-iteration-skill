import sys
import json
from client import MDPValueIterationSolver

def handle_rpc(line):
    try:
        req = json.loads(line)
    except Exception:
        return
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        res = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "genpark-markov-decision-process-value-iteration-skill", "version": "1.0.0"},
            "capabilities": {"tools": {}}
        }
    elif method == "tools/list":
        res = {
            "tools": [
                {
                    "name": "solve_mdp",
                    "description": "Find optimal value function and policy for discrete MDP using Value Iteration",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "states": {"type": "array", "items": {"type": "string"}},
                            "actions": {"type": "array", "items": {"type": "string"}},
                            "transitions": {"type": "object"},
                            "gamma": {"type": "number", "default": 0.95}
                        },
                        "required": ["states", "actions", "transitions"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "solve_mdp":
            raw_t = args.get("transitions", {})
            trans = {}
            for k_str, val in raw_t.items():
                s, a = k_str.split("::") if "::" in k_str else (k_str, "default")
                trans[(s, a)] = val
            solver = MDPValueIterationSolver(
                states=args.get("states", []),
                actions=args.get("actions", []),
                transitions=trans,
                gamma=args.get("gamma", 0.95)
            )
            sol = solver.solve()
            res = {"content": [{"type": "text", "text": json.dumps(sol)}]}
        else:
            res = {"isError": True, "content": [{"type": "text", "text": f"Unknown tool {tool_name}"}]}
    else:
        res = {"error": {"code": -32601, "message": "Method not found"}}

    resp = {"jsonrpc": "2.0", "id": req_id, "result": res.get("result", res)}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        if line.strip():
            handle_rpc(line.strip())

if __name__ == "__main__":
    main()
