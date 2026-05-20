# log.md — AI-Assisted Design Process Record

**Project:** DRL Research Assistant — AI Harness System  
**Course:** Deep Reinforcement Learning — Homework 4  
**Date:** May 2026  
**Tool:** Claude Code (Claude Sonnet 4.6) via VS Code Extension

---

## Overview

This log records the iterative design process of the DRL Research Assistant AI Harness system, including AI-assisted architecture decisions, prompt/chat history, and the evolution from initial idea to final implementation.

---

## Session 1 — Problem Definition and Scope

**Date:** 2026-05-20  
**Prompt to AI:**
> 目前這個專案是我按照之前的作業內容做出來的 但今天作業要求有更新 請你先按照下面的作業規定幫我審查 [新作業規定貼上]

**AI Response Summary:**
The AI performed a gap analysis between the existing DRL survey project (focused on algorithm training) and the new requirement (AI Harness Systems Design). It identified:

- The existing project covered DRL algorithms (DQN, PPO, SAC) with training experiments — completely misaligned with the new topic
- The new homework explicitly requires: LLM as system controller, tool use/function calling, multi-step agent workflow, at least 3 tools, and evaluation design
- The only salvageable content was Part 3 of the existing report (Agentic AI section)

**Key Design Decision #1:**  
The AI proposed reusing the existing SB3 training code as a **tool** within an AI Harness, rather than discarding all prior work. This insight directly shaped the DRL Research Assistant concept.

> "你可以把你已有的程式碼（SB3 訓練腳本）包裝成 tool，讓舊專案的工作不浪費。"

---

## Session 2 — Architecture Design

**Date:** 2026-05-20  
**Prompt to AI:**
> 你剛好提到我想實現的ai agent 就決定是這個主題了 先幫我設計架構並且寫出報告

**Architecture Iteration 1 (Initial Proposal):**

```
User → LLM → Tool Router → [search_papers, run_experiment, analyze]
```

**Problem identified:** The initial flat architecture didn't address memory persistence. If the LLM context window is cleared, all experiment results are lost.

**Architecture Iteration 2 (With Memory Layer):**

```
User → LLM Controller → Tools → [ArXiv API, SB3, JSON DB]
                     ↕
              Memory System
              [Short-term: context | Long-term: experiment_db.json]
```

**Key Design Decision #2 — Memory architecture:**  
The AI suggested splitting memory into two tiers:
- **Short-term** (in-context): conversation history and tool results within a session
- **Long-term** (persistent JSON): experiment results survive session restarts, enabling cross-session analysis

This solved the data loss problem and enabled the `analyze_results(["all"])` feature.

---

## Session 3 — Tool Design Iterations

**Date:** 2026-05-20

### Tool 1: ArXiv Search

**First design (rejected):**  
Initial plan used the `arxiv` Python library (`pip install arxiv`). This added an external dependency.

**Final design:**  
Used Python's built-in `urllib` + `xml.etree.ElementTree` to call the ArXiv Atom API directly — zero extra dependencies, more transparent, easier to audit.

```python
# Rejected (external dep):
import arxiv
search = arxiv.Search(query=query)

# Accepted (stdlib only):
import urllib.request, xml.etree.ElementTree as ET
response = urllib.request.urlopen(base_url + params)
```

**Rationale:** Minimizing dependencies reduces deployment friction and makes the tool more reliable for demo purposes.

### Tool 2: RL Experiment Runner

**Design Challenge:** SB3 training can take minutes. Should the tool be async or synchronous?

**Decision:** Synchronous for simplicity. The LLM naturally waits for the tool result before proceeding. For the demo scope (CartPole with 50k steps ≈ 30 seconds), this is acceptable.

**Key addition:** UUID-based `experiment_id` persisted to `experiment_db.json` before returning, ensuring the result is accessible even if the LLM context is cleared.

### Tool 3: Result Analyzer

**Design Challenge:** How to handle `["all"]` vs. specific IDs?

**Decision:** Special-case `"all"` in the experiment_ids list as a wildcard that loads every experiment from the database. This simplifies the LLM's job — it doesn't need to track all IDs explicitly.

```python
if "all" in experiment_ids:
    selected = list(db.values())
```

**Insight generation:** Added automatic `_generate_insights()` to surface per-environment winners and best/worst runs, reducing the LLM's need to do arithmetic.

---

## Session 4 — Orchestration Design

**Date:** 2026-05-20

**Question posed to AI:**  
What loop mechanism should the agent use?

**Options considered:**

| Option | Pros | Cons |
|--------|------|------|
| Fixed pipeline (search→run→analyze always) | Predictable | Inflexible; wastes calls if results already exist |
| ReAct loop (LLM decides each step) | Flexible, context-aware | Slightly harder to debug |
| LangGraph state machine | Fine-grained control | Over-engineering for 3 tools |

**Decision:** ReAct loop via OpenAI's function calling API. The LLM dynamically decides whether to call another tool based on what's been returned so far. This is the simplest implementation that handles the full range of user queries.

**Key Prompt Engineering Decision:**  
The system prompt encodes the "search first" convention explicitly:

```
Workflow guidelines:
- Always search the literature FIRST when given a new research question.
- Plan the experiment set before calling run_rl_experiment.
- After all runs finish, call analyze_results to produce a ranked comparison.
```

Without this, the LLM sometimes skips the literature search step and goes directly to experiments.

---

## Session 5 — LLM 後端切換：Anthropic → OpenAI

**Date:** 2026-05-20

**Background:** 初始設計使用 Anthropic Claude Sonnet 4.6，但使用者已有可用的 OpenAI API key，因此切換至 GPT-4o。

**API 格式差異對照：**

| 項目 | Anthropic (原始設計) | OpenAI (最終實作) |
|------|---------------------|-----------------|
| 環境變數 | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| 模型 | `claude-sonnet-4-6` | `gpt-4o` |
| Tool 格式 | `{"name": ..., "input_schema": {...}}` | `{"type": "function", "function": {"name": ..., "parameters": {...}}}` |
| 結束條件 | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| Tool 結果 | `role: "user"` + `type: "tool_result"` | `role: "tool"` + `tool_call_id` |
| System prompt | 獨立 `system` 參數 | 放入 messages 第一筆 (`role: "system"`) |

**Impact on codebase:** 只修改 `agent/harness_agent.py`，三個 tool 檔案完全不動。這驗證了工具層與 LLM 層分離的架構優點。

**Removed feature:** Anthropic 的 prompt caching (`cache_control: ephemeral`) 是平台專屬功能，OpenAI 目前不支援相同機制，因此移除。

---

## Session 6 — ArXiv Rate Limit 修正

**Date:** 2026-05-20

**Issue:** GPT-4o 在同一輪回應中連續呼叫 `search_arxiv` 兩次（第一次不帶 `max_results`，第二次帶），觸發 ArXiv API 的 rate limit（HTTP 429），導致後續請求全部 timeout。

**Root cause:** ArXiv 官方要求呼叫間隔至少 3 秒，但連續兩次呼叫間隔約 0 秒。

**Fix:** 在 `arxiv_search.py` 加入固定 3 秒延遲：
```python
time.sleep(3)  # ArXiv rate limit: max 1 req / 3 sec
```

同時將 HTTP 改為 HTTPS，timeout 從 10 秒延長至 30 秒。

---

## Session 7 — Evaluation Design

**Date:** 2026-05-20

**Challenge:** How to evaluate an AI Harness that's not a pure RL agent?

**AI suggestion:** Split evaluation into three dimensions:
1. **Functional correctness** (tool success rates, experiment reproducibility)
2. **Orchestration quality** (does the LLM plan correctly? are tool calls minimal?)
3. **Output quality** (insight accuracy, citation correctness, user satisfaction)

This multi-dimensional approach was adopted directly in the report's evaluation section.

---

## Architecture Decisions Summary

| Decision | Chosen Approach | Alternative Rejected | Reason |
|----------|----------------|---------------------|--------|
| LLM backbone | GPT-4o (OpenAI) | Claude Sonnet 4.6 | User already had OpenAI API key |
| Orchestration | ReAct loop (function calling) | LangChain/LangGraph | Simpler, no extra dependencies |
| ArXiv client | urllib + xml stdlib | arxiv PyPI package | Zero extra dependencies |
| Memory persistence | JSON flat file | SQLite / Redis | Sufficient for demo scale |
| Tool count | 3 (search, run, analyze) | 5+ tools | Minimum sufficient for full workflow |
| Async vs sync | Synchronous | Async (asyncio) | Simplicity; SB3 is CPU-bound anyway |
| ArXiv rate limit | 3s sleep per call | No delay | Prevent HTTP 429 from consecutive calls |

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `agent/harness_agent.py` | Created | Main LLM controller with ReAct loop |
| `agent/tools/arxiv_search.py` | Created | Tool 1: ArXiv paper search |
| `agent/tools/rl_experiment.py` | Created | Tool 2: SB3 experiment runner |
| `agent/tools/result_analyzer.py` | Created | Tool 3: Result comparison |
| `agent/tools/__init__.py` | Created | Tool package init |
| `agent/memory/experiment_db.json` | Auto-created at runtime | Persistent experiment store |
| `docs/report_harness.md` | Created | IEEE format written report |
| `docs/infographic.md` | Created | System architecture diagrams |
| `docs/log.md` | Created | This file |
| `docs/report.md` | Existing | Previous DRL survey (retained for reference) |

---

## Lessons Learned

1. **Reuse before rebuild** — Wrapping existing code (SB3 scripts) as tools was faster than writing new tools from scratch and preserved weeks of prior experiment work.

2. **System prompt as workflow spec** — Encoding the "search before experiment" convention in the system prompt is more reliable than implementing it in Python logic.

3. **Separation of concerns** — Each tool does exactly one thing. The LLM handles composition. This makes each component independently testable.

4. **Persistent memory unlocks multi-session workflows** — Without `experiment_db.json`, every session starts from zero. With it, the agent accumulates knowledge across days of research.
