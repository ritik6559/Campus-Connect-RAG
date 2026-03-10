# JUIT Faculty RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that scrapes faculty information from the JUIT website, stores it in **ChromaDB**, and answers questions using **OpenAI GPT-4o-mini** via **LangChain**.

---

## Architecture

```
JUIT Website (5 dept pages)
      │  BeautifulSoup scraper
      ▼
LangChain Documents  ← 1 doc = 1 faculty member (no splitting)
      │  OpenAIEmbeddings (text-embedding-3-small)
      ▼
ChromaDB  (cosine similarity index, persistent)
      │
      ▼
User Query ──► Query Router
                    │
        ┌───────────┴────────────────┐
   AGGREGATE?                    SEMANTIC?
("list all", "how many"…)    (specific info)
        │                           │
 Metadata filter              top-20 similarity
 fetch ALL docs               search + memory
        │                           │
        └───────────┬───────────────┘
                    │
              GPT-4o-mini
              + Full Context
                    │
                  Answer
```

---

## Project Structure

```
campus-connect-rag/
├── scraper/
│   ├── __init__.py
│   └── faculty_scraper.py      # BeautifulSoup → LangChain Documents
├── vectorstore/
│   ├── __init__.py
│   └── vector_store.py         # Embed, store, aggregate helpers
├── chatbot/
│   ├── __init__.py
│   └── rag_chatbot.py          # Two-path RAG + memory
├── app.py                      # Gradio 6.9 web UI
├── main.py                     # CLI entry point
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Or export it:
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run

**Web UI** (recommended):
```bash
python main.py --ui
# Open http://localhost:7860
# Go to Setup tab → enter API key → Scrape JUIT Website
```

**CLI — scrape live:**
```bash
python main.py --scrape
```

**CLI — save JSON then reuse (faster on reruns):**
```bash
python main.py --scrape --save-json     # scrape once, save to faculty_data.json
python main.py --from-json faculty_data.json   # reload without re-scraping
```

**CLI — use existing ChromaDB (no re-embedding):**
```bash
python main.py --no-ingest
```

---

## Supported Query Types

| Query Type | Examples | How it works |
|---|---|---|
| **Aggregate** | "List all CSE faculty", "How many ECE professors?", "Total faculty count" | Fetches ALL docs via metadata filter — no k-limit |
| **Semantic** | "Who teaches machine learning?", "Email of ECE HOD", "Faculty with VLSI specialization" | Top-20 cosine similarity search + ConversationalRetrievalChain |

---

## Key Design Decisions

**No text splitting** — Faculty profiles are stored as single Documents. Splitting caused aggregate queries to return wrong counts because one person's data was split across chunks.

**Two-path routing** — Aggregate queries bypass semantic search entirely and fetch all matching documents directly from ChromaDB's metadata filter.

**TOP_K = 20** — Raised from 5 to handle most departments without switching to aggregate mode for moderately-sized queries.

---

## Departments Scraped

| Department | URL |
|---|---|
| Computer Science & IT | `/computer-science-engineering-information-technology-faculty` |
| Electronics & Communication | `/electronics-communication-engineering-faculty` |
| Humanities & Social Sciences | `/humanities-social-sciences-faculty` |
| Biotechnology & Informatics | `/biotechnology-and-informatics-faculty` |
| Civil Engineering | `/civil-engineering-faculty` |

---

## LangChain Stack

| Layer | Class |
|---|---|
| LLM | `langchain_openai.ChatOpenAI` (gpt-4o-mini) |
| Embeddings | `langchain_openai.OpenAIEmbeddings` (text-embedding-3-small) |
| Vector Store | `langchain_chroma.Chroma` |
| RAG Chain | `langchain.chains.ConversationalRetrievalChain` |
| Memory | `langchain.memory.ConversationBufferWindowMemory` |
| Prompts | `langchain_core.prompts.ChatPromptTemplate` |
| Documents | `langchain_core.documents.Document` |
| UI | `gradio 6.9` |
