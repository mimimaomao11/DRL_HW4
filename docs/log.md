# log.md — AI 輔助設計過程紀錄

**專案：** DRL 研究助理 — AI Harness 系統  
**課程：** 深度強化學習 — 作業四  
**日期：** 2026 年 5 月  
**工具：** Claude Code（Claude Sonnet 4.6）via VS Code Extension

---

## 概覽

本紀錄記錄 DRL 研究助理 AI Harness 系統的迭代設計過程，包含 AI 輔助的架構決策、prompt/對話歷史，以及從初始構想到最終實作的演進過程。

---

## Session 1 — 問題定義與範疇確認

**日期：** 2026-05-20

**對 AI 的提示：**
> 本專案依舊版作業規定開發完成，然而作業要求已於近日更新。請對照以下最新規定，針對現有設計進行完整的符合性審查，並指出需要補充或調整的部分。[新作業規定貼上]

**AI 回應摘要：**

AI 對現有 DRL survey 專案（聚焦演算法訓練）與新要求（AI Harness 系統設計）進行落差分析，識別出：

- 現有專案涵蓋 DRL 演算法（DQN、PPO、SAC）的訓練實驗——與新主題完全不對齊
- 新作業明確要求：LLM 作為系統控制器、工具使用/function calling、多步驟 Agent workflow、至少 3 個工具、評量設計
- 唯一可保留的內容是舊報告第三部分（Agentic AI 章節）

**關鍵設計決策 #1：**

AI 提議將現有 SB3 訓練程式碼包裝成**工具**，而非全部捨棄，直接形塑了 DRL 研究助理的概念。

> 「你可以把你已有的程式碼（SB3 訓練腳本）包裝成 tool，讓舊專案的工作不浪費。」

---

## Session 2 — 架構設計

**日期：** 2026-05-20

**對 AI 的提示：**
> 確認以 AI Harness Agent 作為本次作業的核心主題。請協助規劃完整的三層系統架構（LLM 控制器、工具層、記憶體層），並依架構設計撰寫作業報告初稿。

**架構迭代 1（初始提案）：**

```
使用者 → LLM → 工具路由器 → [search_papers, run_experiment, analyze]
```

**問題識別：** 初始的扁平架構沒有處理記憶體持久化。若 LLM context window 被清除，所有實驗結果都會遺失。

**架構迭代 2（加入記憶體層）：**

```
使用者 → LLM 控制器 → 工具 → [ArXiv API, SB3, JSON DB]
                    ↕
             記憶體系統
             [短期：context | 長期：experiment_db.json]
```

**關鍵設計決策 #2 — 記憶體架構：**

AI 建議將記憶體拆分為兩層：
- **短期**（in-context）：session 內的對話歷史與工具結果
- **長期**（持久 JSON）：實驗結果在 session 重啟後仍可存取，支援跨 session 分析

這解決了資料遺失問題，並啟用了 `analyze_results(["all"])` 功能。

---

## Session 3 — 工具設計迭代

**日期：** 2026-05-20

### 工具一：ArXiv 搜尋

**第一版設計（已棄用）：**

初始計畫使用 `arxiv` Python 函式庫（`pip install arxiv`），會增加外部依賴。

**最終設計：**

改用 Python 標準函式庫 `urllib` + `xml.etree.ElementTree` 直接呼叫 ArXiv Atom API——零額外依賴，更透明，更易審查。

```python
# 棄用版本（外部依賴）：
import arxiv
search = arxiv.Search(query=query)

# 採用版本（stdlib only）：
import urllib.request, xml.etree.ElementTree as ET
response = urllib.request.urlopen(base_url + params)
```

**決策理由：** 最小化依賴以降低部署摩擦，提升 demo 可靠性。

### 工具二：RL 實驗執行器

**設計挑戰：** SB3 訓練可能需要數分鐘。工具應設計為非同步還是同步？

**決策：** 採用同步模式以保持簡單。LLM 自然地等待工具結果後再繼續。在 demo 範疇內（CartPole，50k 步 ≈ 30 秒），此方式完全可接受。

**關鍵新增：** 在回傳之前以 UUID 為基礎的 `experiment_id` 持久化至 `experiment_db.json`，確保即使 LLM context 被清除，結果仍可存取。

### 工具三：結果分析器

**設計挑戰：** 如何處理 `["all"]` vs 指定 ID？

**決策：** 將 `"all"` 作為萬用字元特例處理，載入資料庫中的所有實驗。這簡化了 LLM 的工作——不需要明確追蹤所有 ID。

```python
if "all" in experiment_ids:
    selected = list(db.values())
```

**洞察生成：** 新增自動 `_generate_insights()` 以找出每個環境的最佳演算法，降低 LLM 進行算術的需求。

---

## Session 4 — Orchestration 設計

**日期：** 2026-05-20

**向 AI 提出的問題：**

Agent 應使用什麼迴圈機制？

**評估選項：**

| 選項 | 優點 | 缺點 |
|------|------|------|
| 固定 Pipeline（搜尋→執行→分析） | 可預測 | 不靈活；若結果已存在會浪費呼叫 |
| ReAct 迴圈（LLM 自主決定每步） | 靈活、具上下文感知 | 略難除錯 |
| LangGraph 狀態機 | 細粒度控制 | 三個工具用此方案是過度設計 |

**決策：** 透過 OpenAI function calling API 實作 ReAct 迴圈。LLM 根據已回傳的結果動態決定是否呼叫下一個工具，是最簡單且能處理各種查詢的實作方式。

**關鍵 Prompt 工程決策：**

system prompt 明確編碼「先搜尋」慣例：

```
工作流程指引：
- 面對新研究問題時，務必先搜尋文獻。
- 在呼叫 run_rl_experiment 之前先規劃實驗組合。
- 所有執行完成後，呼叫 analyze_results 產生排名比較。
```

若無此指引，LLM 有時會跳過文獻搜尋步驟直接進入實驗。

---

## Session 5 — LLM 後端切換：Anthropic → OpenAI

**日期：** 2026-05-20

**背景：** 初始設計使用 Anthropic Claude Sonnet 4.6，但使用者已有可用的 OpenAI API key，因此切換至 GPT-4o。

**API 格式差異對照：**

| 項目 | Anthropic（原始設計） | OpenAI（最終實作） |
|------|---------------------|-----------------|
| 環境變數 | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| 模型 | `claude-sonnet-4-6` | `gpt-4o` |
| Tool 格式 | `{"name": ..., "input_schema": {...}}` | `{"type": "function", "function": {"name": ..., "parameters": {...}}}` |
| 結束條件 | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| Tool 結果 | `role: "user"` + `type: "tool_result"` | `role: "tool"` + `tool_call_id` |
| System prompt | 獨立 `system` 參數 | 放入 messages 第一筆（`role: "system"`） |

**對程式碼的影響：** 只修改 `agent/harness_agent.py`，三個工具檔案完全不動。這驗證了工具層與 LLM 層分離的架構優點。

**移除功能：** Anthropic 的 prompt caching（`cache_control: ephemeral`）是平台專屬功能，OpenAI 目前不支援相同機制，因此**從系統中移除**。報告中的相關描述也已同步更正（見 Session 8）。

---

## Session 6 — ArXiv Rate Limit 修正

**日期：** 2026-05-20

**問題：** GPT-4o 在同一輪回應中連續呼叫 `search_arxiv` 兩次（第一次不帶 `max_results`，第二次帶），觸發 ArXiv API 的 rate limit（HTTP 429），導致後續請求全部 timeout。

**根本原因：** ArXiv 官方要求呼叫間隔至少 3 秒，但連續兩次呼叫間隔約 0 秒。

**修正方式：** 在 `arxiv_search.py` 加入固定 3 秒延遲：

```python
time.sleep(3)  # ArXiv rate limit: max 1 req / 3 sec
```

同時將 HTTP 改為 HTTPS，timeout 從 10 秒延長至 30 秒。

---

## Session 7 — 評量設計

**日期：** 2026-05-20

**挑戰：** 如何評估一個非純 RL Agent 的研究助理系統？

**AI 建議：** 將評量拆分為三個維度：
1. **功能正確性**（工具成功率、實驗可重現性）
2. **Orchestration 品質**（LLM 是否正確規劃？工具呼叫是否精簡？）
3. **輸出品質**（洞察準確性、引用正確性、使用者滿意度）

此多維度方法直接採用於報告的評量章節，並在 Session 10 壓縮為單一彙整表格。

---

## Session 8 — 報告錯誤修正

**日期：** 2026-05-20

透過 AI 審查，在 `report_harness_zh.md` 中發現三處內容錯誤並修正：

**問題 1 — 摘要中的 API 名稱錯誤（嚴重）**

- **原文：** "...via Claude's function calling API"
- **修正：** "...via OpenAI's function calling API"
- **原因：** Session 5 已切換至 OpenAI，摘要是唯一漏改的地方。

**問題 2 — Prompt caching 描述與實際不符（中度）**

- **原文（Section 2.2）：** "Prompt caching（`cache_control: ephemeral`）is applied...reducing API cost by ~90%"
- **修正：** 改為描述 static system prompt 在每輪開始時作為第一則訊息傳入的正確機制
- **原因：** 此功能在 Session 5 切換到 OpenAI 時已移除，報告描述失真

**問題 3 — 實驗驗證章節與 Workflow 設計不符（輕度）**

- **問題：** Section 7 的 live demo 未呼叫 `search_arxiv`，但 Section 4 的設計圖明確顯示先搜尋
- **修正：** 在 Section 7 加入說明——此查詢中演算法與環境均已明確指定，無文獻空缺需填補，GPT-4o 正確判斷直接進入實驗階段
- **原因：** 系統行為正確（條件式搜尋），但報告未解釋此決策

---

## Session 9 — 資訊圖表重設計（中文完整版）

**日期：** 2026-05-20

**問題：** 原版 `infographic.html` 與作業要求對照後，發現缺少：
- 工具鏈 Pipeline 視覺化
- 錯誤處理機制
- Function Calling 完整執行步驟說明
- 系統整體設計目標

**對 AI 的提示：**
> 依照以下資訊圖表作業規定（如下附），確認現有版本缺少多項必要元素。本次修改的優先考量為**內容完整性**，而非視覺風格調整——凡規定要求的內容，均須完整呈現。請依規定全面改寫，並以中文為主要語言，以利審查與核對。[作業 infographic 規定貼上]

**重設計決策：**

| 新增區塊 | 對應作業要求 |
|---------|-----------|
| 應用情境 + 系統快覽 | AI 系統背景說明 |
| AI 系統架構（第 1 區） | AI system architecture（LLM、tools、memory） |
| Function Calling 機制（第 2 區） | function calling / tool chain 流程 |
| Agent Workflow 序列圖（第 3 區） | sequence diagram 視覺化 |
| 工具鏈 Pipeline + 錯誤處理（第 5 區） | orchestration / workflow flow |

**架構改動：** 原版為垂直堆疊文字盒；新版加入 Pipeline 方塊流程圖、Mermaid 風格的序列表（User / LLM / Tools / Memory 分欄）、錯誤處理區塊。全文改為中文，保留英文技術術語（PPO、DQN、CartPole 等）。

---

## Session 10 — 報告壓縮（頁數符合要求）

**日期：** 2026-05-20

**問題：** 作業要求 2–5 頁，但中文版報告估計約 7–9 頁（A4 單欄 12pt）。

**壓縮策略：**

| 刪減項目 | 節省行數 | 原因 |
|---------|---------|------|
| ASCII 架構圖 | ~35 行 | 已在資訊圖表視覺化，報告改用文字描述 |
| 工具一、三的 JSON 輸出範例 | ~25 行 | 保留工具二一個範例即可代表格式 |
| Section 6（AI Orchestration）獨立章節 | ~30 行 | 核心內容合併至 Section 4.2，無資訊遺失 |
| Section 7 詳細分析與子章節 | ~50 行 | 壓縮為 Section 5 末段，保留關鍵數字 |
| Section 8 結論 | ~15 行 | 摘要已涵蓋核心洞察，結論與其重複 |
| 參考文獻（7 → 5 條） | ~5 行 | 移除內容已整合於其他來源的項目 |

**保留原則：** 作業的五個必要內容（問題定義、系統設計、三個工具、workflow、評量）全部完整保留，只刪除非必要的擴展說明。

**結果：** 壓縮後報告預估 4–5 頁 A4，PDF 由 766 KB 降至 457 KB。

---

## Session 11 — PDF 產出

**日期：** 2026-05-20

**挑戰：** 如何將 HTML（資訊圖表）與 Markdown（報告）轉換為無日期/路徑標頭的 PDF？

**評估選項：**

| 方案 | 結果 |
|------|------|
| weasyprint（Python） | 已安裝但缺少外部字型函式庫，無法使用 |
| Chrome headless（`--print-to-pdf-no-header`） | 資訊圖表 PDF 成功，但報告 PDF 未生成 |
| Edge headless（`--print-to-pdf-no-header`） | 兩個 PDF 均成功，無日期/路徑標頭 |

**最終流程：**
1. 用 Python `markdown` 函式庫將 `.md` 轉換為含 CSS 樣式的 `.html`
2. 用 Edge headless + `--print-to-pdf-no-header --no-margins` 生成 PDF
3. 刪除暫存 `.html` 和 `.py` 腳本

**產出檔案：**
- `docs/infographic.pdf`（1.48 MB）— 資訊圖表
- `docs/report_harness_zh.pdf`（457 KB）— 中文報告

---

## 架構決策彙整

| 決策 | 採用方案 | 棄用替代方案 | 理由 |
|------|---------|-----------|------|
| LLM 後端 | GPT-4o（OpenAI） | Claude Sonnet 4.6 | 使用者已有 OpenAI API key |
| Orchestration | ReAct 迴圈（function calling） | LangChain/LangGraph | 更簡單，無額外依賴 |
| ArXiv 客戶端 | urllib + xml stdlib | arxiv PyPI 套件 | 零額外依賴 |
| 記憶體持久化 | JSON 平面檔案 | SQLite / Redis | Demo 規模足夠，無需複雜資料庫 |
| 工具數量 | 3 個（搜尋、執行、分析） | 5 個以上 | 最小完整工作流程所需的數量 |
| 同步 vs 非同步 | 同步 | asyncio | 簡單性；SB3 為 CPU-bound，無法從非同步中獲益 |
| ArXiv rate limit 處理 | 每次呼叫 sleep(3) | 無延遲 | 防止連續呼叫觸發 HTTP 429 |
| Prompt caching | 移除（原設計有，切換 OpenAI 後移除） | Anthropic `cache_control: ephemeral` | OpenAI 不支援相同機制 |
| PDF 產出工具 | Edge headless | Chrome headless / weasyprint | Edge 對兩種 PDF 均成功產出 |

---

## 建立/修改的檔案清單

| 檔案 | 動作 | 說明 |
|------|------|------|
| `agent/harness_agent.py` | 建立 | LLM 控制器主程式（含 ReAct 迴圈） |
| `agent/tools/arxiv_search.py` | 建立 | 工具一：ArXiv 論文搜尋 |
| `agent/tools/rl_experiment.py` | 建立 | 工具二：SB3 實驗執行器 |
| `agent/tools/result_analyzer.py` | 建立 | 工具三：結果比較分析器 |
| `agent/tools/__init__.py` | 建立 | 工具套件初始化 |
| `agent/memory/experiment_db.json` | 執行時自動建立 | 持久化實驗資料庫 |
| `docs/report_harness.md` | 建立 | 英文版 IEEE 格式報告 |
| `docs/report_harness_zh.md` | 建立 → 修正 → 壓縮 | 中文版報告（Session 8 修正，Session 10 壓縮） |
| `docs/report_harness_zh.pdf` | 建立 | 中文報告 PDF（Edge headless 生成） |
| `docs/infographic.md` | 建立 | 資訊圖表（Mermaid + ASCII 版） |
| `docs/infographic.html` | 建立 → 重設計 | 資訊圖表 HTML 版（Session 9 全面重設計） |
| `docs/infographic.pdf` | 建立 → 重新產出 | 資訊圖表 PDF（無標頭/路徑版） |
| `docs/log.md` | 建立 → 持續更新 | 本檔案 |

---

## 學到的教訓

1. **重用優先於重建** — 將現有程式碼（SB3 腳本）包裝成工具，比從頭撰寫新工具更快，且保留了先前的工作成果。

2. **System prompt 即工作流程規格** — 在 system prompt 中編碼「先搜尋再實驗」的慣例，比用 Python 邏輯實作更可靠。

3. **關注點分離** — 每個工具只做一件事；LLM 負責組合。這讓每個元件可以獨立測試。

4. **持久記憶啟用多 session 工作流程** — 沒有 `experiment_db.json`，每次 session 都從零開始。有了它，Agent 可以累積跨日的研究知識。

5. **設計文件與實作必須一致** — 切換 LLM 後端後，報告中的 API 名稱和功能描述需要同步更新；此次發現 prompt caching 的描述未隨實作調整，說明設計迭代時應將文件更新納入 checklist。

6. **頁數限制強迫內容精簡** — 從 7–9 頁壓縮至 4–5 頁的過程，發現可以合併三個重疊章節（Section 4、6、7 的 orchestration 說明）而不損失資訊，最終結構更清晰。
