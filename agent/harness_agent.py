"""DRL Research Assistant — AI Harness Agent using Claude function calling (tool use)."""

import json
import os
import anthropic
from tools import search_arxiv, run_rl_experiment, analyze_results

# ── Tool schemas (function calling definitions) ────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "search_arxiv",
        "description": (
            "Search ArXiv for research papers on DRL / AI topics. "
            "Use this first to understand the state-of-the-art before designing experiments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'proximal policy optimization robotics'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum papers to return (default 5, max 20)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_rl_experiment",
        "description": (
            "Train a Stable-Baselines3 RL agent on a Gymnasium environment. "
            "Results are saved to a persistent database and can be retrieved later."
        ),
        "input_schema": {
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
    {
        "name": "analyze_results",
        "description": (
            "Load and compare experiment results from the database. "
            "Pass experiment IDs from previous run_rl_experiment calls, or [\"all\"] for every run."
        ),
        "input_schema": {
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
]

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are DRL-Agent, an AI research assistant specialising in Deep Reinforcement Learning.

Your capabilities:
1. search_arxiv   — find relevant research papers before designing experiments
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
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Send a message and run the ReAct tool-use loop until a final text reply."""
        self.history.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=[
                    # Cache the static system prompt across turns
                    {"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}},
                ],
                tools=TOOLS,
                messages=self.history,
            )

            # Collect text blocks for display while building the assistant turn
            assistant_turn_content = list(response.content)
            self.history.append({"role": "assistant", "content": assistant_turn_content})

            if response.stop_reason != "tool_use":
                # Final text reply
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                return "\n".join(text_blocks)

            # Execute all requested tools and feed results back
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                    output = _dispatch(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })

            self.history.append({"role": "user", "content": tool_results})

    def reset(self) -> None:
        """Clear conversation history for a fresh session."""
        self.history = []


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  DRL Research Assistant Agent")
    print("  Type 'exit' to quit | 'reset' to clear history")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[ERROR] Set ANTHROPIC_API_KEY environment variable first.\n")
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
