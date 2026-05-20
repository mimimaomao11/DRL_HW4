# DRL Research Assistant: An AI Harness System for Automated Literature Review and Experiment Management

**Course:** Deep Reinforcement Learning — Homework 4  
**Date:** May 2026  
**Format:** IEEE Style Report

---

## Abstract

This report presents the design and implementation of a **DRL Research Assistant**, an AI Harness system that positions a Large Language Model (LLM) as the central controller for an automated deep reinforcement learning research pipeline. The system addresses the repetitive and time-intensive bottlenecks in DRL research—namely, literature discovery, experiment configuration, and multi-run comparison—by integrating three purpose-built tools via Claude's function calling API. The agent executes a ReAct (Reason + Act) workflow loop, chaining tool calls based on dynamic reasoning until a research question is fully resolved. We describe the system architecture, tool specifications, orchestration logic, and evaluation methodology.

---

## 1. Problem Definition and Application Background

### 1.1 Research Context

A DRL researcher investigating a new task (e.g., "Is PPO or SAC better for lunar landing?") must currently perform three costly manual steps:

1. **Literature search** — browse ArXiv, read abstracts, and synthesize findings
2. **Experiment design and execution** — configure SB3 training scripts, manage hyperparameters, and wait for runs to finish
3. **Result analysis** — aggregate reward curves, compute statistics, and draw conclusions

Each step is largely procedural and does not require creative scientific judgment. This creates a strong motivation for an **AI Harness** that automates the pipeline, allowing the researcher to focus on higher-level hypothesis generation.

### 1.2 Application Scenario

**User:** A DRL graduate student or researcher  
**Goal:** Answer research questions about algorithm comparison, environment difficulty, and hyperparameter sensitivity using an AI agent  
**Input:** Natural language queries (e.g., *"Compare PPO and DQN on CartPole with 50k steps"*)  
**Output:** Structured reports with paper citations, experiment results, ranked comparisons, and actionable insights

---

## 2. AI System Architecture

The system follows a three-layer architecture: **Controller → Tools → Memory**.

```
┌──────────────────────────────────────────────────────────────┐
│                        User Interface (CLI)                   │
│                   Natural Language Query Input                │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                 LLM Controller  (GPT-4o)                      │
│                                                               │
│  • Interprets user intent                                     │
│  • Plans tool-call sequence (ReAct reasoning)                 │
│  • Synthesizes tool outputs into final response               │
│  • Maintains conversation context across turns                │
└──────┬──────────────────────┬──────────────────────┬─────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Tool 1    │    │     Tool 2       │    │    Tool 3       │
│search_arxiv │    │run_rl_experiment │    │analyze_results  │
│             │    │                  │    │                 │
│ ArXiv API   │    │ Stable-Baselines3│    │ experiment_db   │
│ (HTTP/XML)  │    │ + Gymnasium      │    │ (JSON)          │
└──────┬──────┘    └────────┬─────────┘    └───────┬─────────┘
       │                    │                       │
       └────────────────────┴───────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                       Memory System                           │
│                                                               │
│  Short-term: Conversation history (LLM context window)        │
│  Long-term:  experiment_db.json  (persistent result store)    │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 LLM as System Controller

The LLM (GPT-4o) acts as an autonomous **orchestrator**, not just a text generator. It:

- Reads the user's natural language query and infers research intent
- Decides **which tools to call**, **in what order**, and **with what parameters**
- Evaluates each tool's output and decides whether another tool call is needed
- Synthesizes all intermediate results into a coherent final response

The LLM's system prompt encodes domain knowledge (DRL algorithms, Gymnasium environments) and workflow guidelines (search first, then experiment, then analyze).

### 2.2 Memory System

| Memory Type | Implementation | Contents |
|-------------|---------------|---------|
| Working memory | LLM context window (in-context) | Conversation turns, tool results |
| Short-term episodic | `self.history` list in Python | Multi-turn chat history |
| Long-term persistent | `agent/memory/experiment_db.json` | Experiment results, rewards, timestamps |

Prompt caching (`cache_control: ephemeral`) is applied to the static system prompt, reducing API cost on repeated turns by ~90%.

---

## 3. Tool Design

### Tool 1: `search_arxiv(query, max_results)`

| Property | Detail |
|----------|--------|
| **Purpose** | Retrieve relevant research papers from ArXiv |
| **API** | ArXiv Atom Feed API (HTTP GET, no authentication required) |
| **Input** | `query: str` — free-text search; `max_results: int` (default 5) |
| **Output** | `{papers: [{title, authors, abstract, url, published}], count, query}` |
| **When called** | First step in any research workflow; provides literature grounding |

**Example call:**
```json
{ "query": "SAC soft actor-critic continuous control", "max_results": 3 }
```

**Example output (truncated):**
```json
{
  "papers": [
    {
      "title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning",
      "authors": ["Tuomas Haarnoja", "Aurick Zhou", "Pieter Abbeel"],
      "abstract": "We present soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy framework...",
      "url": "https://arxiv.org/abs/1801.01290",
      "published": "2018-01-04"
    }
  ],
  "count": 3
}
```

---

### Tool 2: `run_rl_experiment(algorithm, environment, n_steps, hyperparams)`

| Property | Detail |
|----------|--------|
| **Purpose** | Train an SB3 agent and persist results |
| **Library** | Stable-Baselines3 + Gymnasium |
| **Input** | `algorithm: "DQN"|"PPO"|"SAC"`, `environment: str`, `n_steps: int`, `hyperparams: dict` |
| **Output** | `{experiment_id, algorithm, environment, mean_reward, std_reward, status, timestamp}` |
| **When called** | After literature review; agent designs the experiment set autonomously |

**Example call:**
```json
{
  "algorithm": "PPO",
  "environment": "CartPole-v1",
  "n_steps": 50000,
  "hyperparams": { "learning_rate": 3e-4, "clip_range": 0.2 }
}
```

**Side effect:** Result is saved to `experiment_db.json` under a UUID-based `experiment_id`, enabling retrieval by Tool 3 in a later turn.

---

### Tool 3: `analyze_results(experiment_ids, metrics)`

| Property | Detail |
|----------|--------|
| **Purpose** | Load stored experiments, rank by performance, and generate insights |
| **Data source** | `agent/memory/experiment_db.json` |
| **Input** | `experiment_ids: list[str]` (specific IDs or `["all"]`); `metrics: list[str]` |
| **Output** | `{comparison_table (sorted), best_experiment, insights: list[str]}` |
| **When called** | After all `run_rl_experiment` calls complete |

**Example output:**
```json
{
  "best_experiment": { "algorithm": "PPO", "environment": "CartPole-v1", "mean_reward": 487.3 },
  "comparison_table": [
    { "algorithm": "PPO", "mean_reward": 487.3, "std_reward": 12.4 },
    { "algorithm": "DQN", "mean_reward": 144.1, "std_reward": 67.2 }
  ],
  "insights": [
    "Best run: PPO on CartPole-v1 — mean reward 487.3 ± 12.4",
    "On CartPole-v1: best algorithm is PPO (487.3)"
  ]
}
```

---

## 4. Agent Workflow — Multi-Step Task Execution

The agent follows a **ReAct (Reason + Act)** loop: the LLM alternates between reasoning (producing text) and acting (calling a tool), continuing until `stop_reason == "end_turn"`.

### 4.1 Workflow Diagram

```
User: "Compare PPO and DQN on CartPole with 50k steps"
         │
         ▼
   [LLM Reasoning]
   "I should first search for papers to understand
    the algorithms, then run both experiments."
         │
         ▼
   [Tool Call 1: search_arxiv]
   query = "DQN PPO CartPole comparison"
         │  ←── ArXiv API response (3 papers)
         ▼
   [LLM Reasoning]
   "Papers found. Now run PPO experiment."
         │
         ▼
   [Tool Call 2: run_rl_experiment]
   algorithm=PPO, environment=CartPole-v1, n_steps=50000
         │  ←── {experiment_id: "a1b2c3d4", mean_reward: 487.3}
         ▼
   [Tool Call 3: run_rl_experiment]
   algorithm=DQN, environment=CartPole-v1, n_steps=50000
         │  ←── {experiment_id: "e5f6g7h8", mean_reward: 144.1}
         ▼
   [Tool Call 4: analyze_results]
   experiment_ids = ["a1b2c3d4", "e5f6g7h8"]
         │  ←── comparison table + insights
         ▼
   [LLM Synthesis]
   Produces final markdown report with citations,
   comparison table, and algorithm recommendations.
         │
         ▼
   Agent response to user
```

### 4.2 Key Orchestration Properties

| Property | Mechanism |
|----------|----------|
| **Tool sequencing** | LLM decides order; search before experiment by convention in system prompt |
| **Error recovery** | If a tool returns `"status": "failed"`, LLM diagnoses and retries with adjusted parameters |
| **Multi-turn memory** | `self.history` accumulates all turns; LLM can reference past experiment IDs |
| **Parallel experiments** | LLM can call `run_rl_experiment` multiple times in sequence for different algorithms |

---

## 5. Evaluation Method

Since this is a research-assistant agent (not a pure RL task), evaluation requires multiple complementary metrics.

### 5.1 Functional Correctness

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| Tool call success rate | Fraction of tool calls returning `status: "completed"` | ≥ 95% |
| Experiment reproducibility | Std deviation across 3 identical experiment runs | ≤ 10% of mean |
| Paper retrieval relevance | Human rating (1–5) of ArXiv results vs. query intent | ≥ 4.0 |

### 5.2 Orchestration Quality

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| Average tool calls per query | Count across 20 benchmark queries | 3–5 (optimal range) |
| Plan correctness | Does the LLM search before experimenting? | ≥ 90% of queries |
| Unnecessary tool calls | Redundant calls without new information | ≤ 1 per session |

### 5.3 Output Quality

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| Insight accuracy | Cross-reference agent insights with ground-truth experiment rankings | ≥ 90% match |
| Citation accuracy | Verify paper titles/authors against ArXiv ground truth | ≥ 95% correct |
| User satisfaction | Survey score (1–5) on 10 real research queries | ≥ 4.0 |

### 5.4 System Performance

| Metric | Measurement | Target |
|--------|-------------|--------|
| End-to-end latency (no training) | Search + analyze query | ≤ 10 seconds |
| API cost per session | Tokens billed (with prompt caching) | ≤ $0.05/session |
| Cache hit rate | Fraction of system prompt tokens served from cache | ≥ 80% |

---

## 6. AI Orchestration — Decision Flow and Control

### 6.1 Function Calling Mechanism

OpenAI's **function calling API** is the orchestration backbone. Each tool is defined as a JSON schema with `type: "function"` wrapping a `name`, `description`, and `parameters` object. The LLM selects tools by returning `finish_reason = "tool_calls"` with a list of `tool_calls` objects. The Python harness dispatches each call to the actual function implementation and returns results as `role: "tool"` messages keyed by `tool_call_id`.

```
LLM output:  finish_reason="tool_calls"
             → message.tool_calls: [{id, function.name, function.arguments}]
Python:      → _dispatch(name, json.loads(arguments))
             → returns JSON string
Python:      → appends {"role": "tool", "tool_call_id": id, "content": result}
LLM input:   → next chat.completions.create() call with full history
```

### 6.2 Decision-Making Logic

The LLM makes three classes of decisions autonomously:

1. **Tool selection** — Which tool to call based on current context (literature gap → search; experiment gap → run; comparison needed → analyze)
2. **Parameter inference** — Infers `n_steps`, `hyperparams`, and `environment` from user intent and literature findings
3. **Termination** — Decides when enough information exists to synthesize a final response (no more tool calls)

### 6.3 Memory-Driven Continuity

Cross-turn experiment references are possible because `experiment_db.json` persists results independently of the LLM context window. The user can ask "analyze the runs from last session" in a new conversation, and `analyze_results(["all"])` will load all historical data.

---

## 7. Conclusion

The DRL Research Assistant demonstrates that an LLM with three narrowly scoped tools can automate a non-trivial research workflow end-to-end. Key design insights:

1. **LLM as controller, not just generator** — The value comes from the LLM's ability to plan tool sequences, not just produce text.
2. **Minimal tool surface, maximum composability** — Three tools cover search, execution, and analysis; complex workflows emerge from their composition.
3. **Persistent memory decouples sessions** — The JSON experiment database allows cross-session analysis without re-running experiments.
4. **Prompt caching reduces cost** — Static system prompt caching cuts per-turn API cost significantly in multi-turn sessions.

Future extensions include adding a `plot_results` tool for automatic figure generation, integrating WandB for experiment tracking, and expanding the algorithm set to TD3 and A2C.

---

## References

1. Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
2. Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning. *ICML*.
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
4. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
5. Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*.
6. OpenAI (2025). Function calling. *OpenAI Platform Documentation*.
7. Chase, H. (2022). LangChain — Building applications with LLMs through composability. *GitHub*.
