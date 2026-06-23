# MAD Fact-Checking

A **Multi-Agent Debate** system for automated fact-checking, built for the [FEVER 8 Shared Task](https://fever.ai/task.html) on the AVeriTeC benchmark.

Given a claim, three LangGraph agents debate it in rounds — a **Proponent** searches for supporting evidence, an **Opponent** searches for refuting evidence, and a **Judge** evaluates both sides and delivers a verdict. The system supports two modes:

- **Demo mode** — live web search via Tavily, Vietnamese-language UI, no data needed
- **Evaluation mode** — searches a local AVeriTeC knowledge store, English-language, for batch scoring

Verdicts follow the AVeriTeC schema: `Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`.

---

## Project structure

```
mad_fact_checking/
├── app/
│   ├── backend/             # LangGraph agent server (Python)
│   │   ├── app/
│   │   │   ├── core/        # Graph definition, prompts, tools
│   │   │   ├── schemas/     # Pydantic state & output models
│   │   │   └── services/    # LLM, retrieval, reranking, web search
│   │   └── langgraph.json   # LangGraph deployment config
│   └── frontend/            # Next.js 15 chat UI
├── eval.py                  # Batch evaluation runner (AVeriTeC test/dev)
├── rerun_false.py           # Targeted rerun for wrong predictions
├── convert_evidence.py      # Convert JSONL results → AVeriTeC submission CSV
├── analyze.py               # LLM-based error categorisation tool
├── download_data.sh         # Download AVeriTeC splits & knowledge store
└── requirements.txt         # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ and [pnpm](https://pnpm.io/installation)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Tavily API key](https://tavily.com) *(demo/web mode only)*

### 1 — Python dependencies

```bash
pip install -r requirements.txt
```

This installs everything needed for both the LangGraph backend and the evaluation scripts.

### 2 — Backend environment variables

Create `app/backend/.env`:

```env
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...        # only needed for demo/web mode

# Optional overrides (defaults shown)
DEFAULT_LLM_MODEL=gpt-4.1-mini
MAX_ROUND_DEBATE=3
KNOWLEDGE_STORE_DIR=../../knowledge_store/dev/
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
DEMO_EMBEDDING_MODEL=dangvantuan/vietnamese-document-embedding
TOP_K_RETRIEVAL=1000
TOP_K_RERANKING=10
```

### 3 — Frontend dependencies

```bash
cd app/frontend
npm install -g pnpm
pnpm install
```

### 4 — AVeriTeC data *(evaluation mode only)*

```bash
bash download_data.sh
```

Downloads the AVeriTeC train/dev/test splits and the knowledge store to `knowledge_store/` and `data_store/`.

---

## Running

Open two terminals from the project root.

**Terminal 1 — Backend**

```bash
cd app/backend
langgraph dev
# Server starts at http://localhost:2024
```

**Terminal 2 — Frontend**

```bash
cd app/frontend
pnpm dev
# UI available at http://localhost:3000
```

---

## Usage

### Demo mode (Vietnamese, web search)

Open `http://localhost:3000` and type any claim in Vietnamese. The system searches the web in real time via Tavily and returns a verdict in under 60 seconds.

```
Bộ Y tế Việt Nam đã phê duyệt vaccine COVID-19 của AstraZeneca vào tháng 2 năm 2021.
```

### Evaluation mode (English, knowledge store)

Prefix the claim with its AVeriTeC index to trigger knowledge-store retrieval:

```
42. The Eiffel Tower is located in Berlin.
```

### Batch evaluation (AVeriTeC test set)

```bash
# Run the full pipeline on claims 0–499
python eval.py --start 0 --end 500

# Re-run only previously wrong predictions
python rerun_false.py

# Convert JSONL results to AVeriTeC submission CSV
python convert_evidence.py
```

Results are written to `results.jsonl`. Each entry contains the claim index, predicted verdict, justification, and all retrieved evidence.

---

## Architecture

```
User claim
    │
    ▼
init_debate
    │
    ├──▶ proponent_agent  (HyDE / query gen → Tavily/BM25 → bi-encoder rerank → reasoning)
    │                                                  │
    └──▶ opponent_agent   (same pipeline, opposite stance)
                                                       │
                                                 judge_agent
                                                /           \
                                           RESOLVE          RETRY
                                              │               │
                                           verdict     history_summarizer
                                                             │
                                                      next round ↑
```

The evidence retrieval pipeline for each debater:

1. **Query generation** — HyDE excerpts (round 0) or search queries (later rounds)
2. **Retrieval** — BM25 over the knowledge store *or* parallel Tavily web search
3. **Reranking** — bi-encoder reranker selects the top-10 most relevant passages
4. **Reasoning** — structured LLM call produces verdict + justification + cited evidence tags
