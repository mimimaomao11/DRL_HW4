"""DRL Research Assistant — AI Harness Agent using OpenAI function calling (tool use)."""

import json
import os
from openai import OpenAI
from tools import search_arxiv, run_rl_experiment, analyze_results

# ── Tool schemas (OpenAI function calling format) ──────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search ArXiv for research papers on DRL / AI topics. "
                "Use this first to understand the state-of-the-art before designing experiments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'proximal policy optimization robotics'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum papers to return (default 5, max 20)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rl_experiment",
            "description": (
                "Train a Stable-Baselines3 RL agent on a Gymnasium environment. "
                "Results are saved to a persistent database and can be retrieved later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["DQN", "PPO", "SAC"],
                        "description": "RL algorithm to use",
                    },
                    "environment": {
                        "type": "string",
                        "description": "Gymnasium environment ID, e.g. 'CartPole-v1'",
                    },
                    "n_steps": {
                        "type": "integer",
                        "description": "Total training timesteps",
                    },
                    "hyperparams": {
                        "type": "object",
                        "description": "Optional SB3 algorithm kwargs, e.g. {\"learning_rate\": 3e-4}",
                    },
                },
                "required": ["algorithm", "environment", "n_steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_results",
            "description": (
                "Load and compare experiment results from the database. "
                "Pass experiment IDs from previous run_rl_experiment calls, or [\"all\"] for every run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of experiment IDs, or [\"all\"] to compare every run",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to include (default: [\"mean_reward\", \"std_reward\"])",
                    },
                },
                "required": ["experiment_ids"],
            },
        },
    },
]

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are DRL-Agent, an AI research assistant specialising in Deep Reinforcement Learning.

Your capabilities:
1. search_arxiv      — find relevant research papers before designing experiments
2. run_rl_experiment — execute training runs with SB3 (DQN / PPO / SAC)
3. analyze_results   — compare stored experiments and surface insights

Workflow guidelines:
- Always search the literature FIRST when given a new research question.
- Plan the experiment set before calling run_rl_experiment (explain choices).
- After all runs finish, call analyze_results to produce a ranked comparison.
- Cite paper titles and authors when discussing algorithms.
- Keep responses concise; use markdown tables for comparisons.
- If an experiment fails, diagnose the cause and suggest a fix.
"""

# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _dispatch(name: str, params: dict) -> str:
    """Execute the named tool and return its result as a JSON string."""
    if name == "search_arxiv":
        result = search_arxiv(**params)
    elif name == "run_rl_experiment":
        result = run_rl_experiment(**params)
    elif name == "analyze_results":
        result = analyze_results(**params)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Agent class ────────────────────────────────────────────────────────────────

class DRLResearchAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str) -> str:
        """Send a message and run the ReAct tool-use loop until a final text reply."""
        self.history.append({"role": "user", "content": user_message})

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                tools=TOOLS,
                messages=self.history,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Add assistant message to history (includes tool_calls if any)
            self.history.append(message)

            if finish_reason != "tool_calls":
                return message.content or ""

            # Execute all requested tool calls and feed results back
            for tool_call in message.tool_calls:
                params = json.loads(tool_call.function.arguments)
                print(f"  [tool] {tool_call.function.name}({json.dumps(params, ensure_ascii=False)})")
                output = _dispatch(tool_call.function.name, params)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })

    def reset(self) -> None:
        """Clear conversation history (keep system prompt)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  DRL Research Assistant Agent  (OpenAI)")
    print("  Type 'exit' to quit | 'reset' to clear history")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[ERROR] Set OPENAI_API_KEY environment variable first.\n")
        return

    agent = DRLResearchAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[History cleared]")
            continue

        print("\nAgent: ", end="", flush=True)
        reply = agent.chat(user_input)
        print(reply)


if __name__ == "__main__":
    main()
