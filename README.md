# HW4 — DRL Research Assistant: AI Harness System

> **Course:** Deep Reinforcement Learning — Homework 4  
> **Topic:** AI Harness Systems Design and Analysis  
> **Date:** May 2026

---

## What is this?

A **DRL Research Assistant** powered by GPT-4o (function calling).  
The agent automates three repetitive bottlenecks in DRL research:

1. **Literature search** — query ArXiv and get structured paper summaries
2. **Experiment execution** — train DQN / PPO / SAC agents via Stable-Baselines3
3. **Result analysis** — compare multiple runs with ranked insights

The LLM acts as the **system controller**, dynamically deciding which tool to call based on the user's natural language query (ReAct loop). Results persist across sessions via a local JSON database.

---

## Project Structure

```
HW4/
├── README.md
│
├── agent/
│   ├── harness_agent.py          ← Main agent: LLM controller + ReAct loop
│   ├── tools/
│   │   ├── arxiv_search.py       ← Tool 1: ArXiv paper search (stdlib, no extra deps)
│   │   ├── rl_experiment.py      ← Tool 2: SB3 RL training + result persistence
│   │   └── result_analyzer.py    ← Tool 3: Experiment comparison + insight generation
│   └── memory/
│       └── experiment_db.json    ← Auto-created: persistent experiment results
│
├── docs/
│   ├── report_harness.md         ← Written report (IEEE format, ~4 pages)
│   ├── infographic.md            ← System architecture diagrams (ASCII + Mermaid)
│   └── log.md                    ← AI-assisted design process log
│
└── drl_env/                      ← Python virtual environment
```

---

## Quick Start

### 1. Install dependencies

```bash
# Activate virtual environment (Windows)
drl_env\Scripts\activate

# Install required packages
pip install openai stable-baselines3 gymnasium
```

### 2. Set API key

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."

# Linux / Mac
export OPENAI_API_KEY="sk-..."
```

### 3. Run the agent

```bash
python agent/harness_agent.py
```

---

## Example Session

```
============================================================
  DRL Research Assistant Agent
  Type 'exit' to quit | 'reset' to clear history
============================================================

You: Compare PPO and DQN on CartPole with 50k steps

  [tool] search_arxiv({"query": "DQN PPO CartPole comparison"})
  [tool] run_rl_experiment({"algorithm": "PPO", "environment": "CartPole-v1", "n_steps": 50000})
  [tool] run_rl_experiment({"algorithm": "DQN", "environment": "CartPole-v1", "n_steps": 50000})
  [tool] analyze_results({"experiment_ids": ["a1b2c3d4", "e5f6g7h8"]})

Agent: ## Experiment Results

| Algorithm | Environment  | Mean Reward | Std  |
|-----------|-------------|-------------|------|
| **PPO**   | CartPole-v1 | **487.3**   | 12.4 |
| DQN       | CartPole-v1 | 144.1       | 67.2 |

**Finding:** PPO outperforms DQN by 3.4× on CartPole-v1 (50k steps).
PPO's clipped surrogate objective prevents catastrophic forgetting,
which is the dominant failure mode for DQN in this task.

**Literature:** Schulman et al. (2017) show that PPO's constraint
on policy update magnitude is key to its stability advantage.
```

---

## System Architecture

```
 User (natural language query)
          │
          ▼
 ┌─────────────────────────────────────────┐
 │          LLM Controller (GPT-4o)         │
 │  • Parse intent                         │
 │  • Plan tool sequence (ReAct reasoning) │
 │  • Synthesize results into response     │
 └──────┬──────────────┬───────────────────┘
        │              │              │
        ▼              ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌──────────────┐
  │  Tool 1  │  │   Tool 2   │  │   Tool 3     │
  │  search_ │  │  run_rl_   │  │  analyze_    │
  │  arxiv() │  │ experiment │  │  results()   │
  │          │  │    ()      │  │              │
  │ ArXiv API│  │ SB3 +      │  │ experiment_  │
  │ (stdlib) │  │ Gymnasium  │  │ db.json      │
  └──────────┘  └────────────┘  └──────────────┘
                      │
                      ▼
         ┌─────────────────────────────┐
         │        Memory System        │
         │                             │
         │ Short-term: context window  │
         │ Long-term:  experiment_db   │
         └─────────────────────────────┘
```

---

## Tools Reference

| Tool | Input | Output | External Dependency |
|------|-------|--------|---------------------|
| `search_arxiv(query, max_results)` | Free-text query | `{papers[], count}` | None (stdlib) |
| `run_rl_experiment(algorithm, environment, n_steps, hyperparams)` | SB3 config | `{experiment_id, mean_reward, std_reward, status}` | `stable-baselines3`, `gymnasium` |
| `analyze_results(experiment_ids, metrics)` | List of IDs or `["all"]` | `{comparison_table[], best_experiment, insights[]}` | None |

---

## Deliverables

| File | Description |
|------|-------------|
| [docs/report_harness.md](docs/report_harness.md) | Written report — problem definition, architecture, tool design, workflow, evaluation, orchestration |
| [docs/infographic.md](docs/infographic.md) | Visual system design — ASCII architecture, Mermaid sequence diagram, tool I/O summary |
| [docs/log.md](docs/log.md) | Design process log — 6 sessions, all architecture decisions and rejected alternatives |

---

## Environment

| Item | Version |
|------|---------|
| Python | 3.13 |
| openai | 2.37.0 |
| stable-baselines3 | 2.8.0 |
| gymnasium | 1.2.3 |
| OS | Windows 11 |
