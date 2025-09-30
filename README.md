# AI Tinkerers Hackathon Project: Research Assistant

## Project Description

This project is a **Streamlit-based research assistant** that searches
arXiv and Zotero via MCP tools, summarizes and organizes results, and
lets users like/dislike papers to rerank recommendations. It
demonstrates an **end-to-end agentic workflow** with Google's ADK,
Gemini models, tool calling over MCP, and **human-in-the-loop feedback**
that influences retrieval ranking.

Additional features include **gap analysis**, **literature review
generation**, and the ability to **export liked papers directly to
Zotero**.

------------------------------------------------------------------------

## Technical Excellence

### End-to-End Demo

1. User enters a research topic (or starts from a Zotero collection)\
2. An ADK `LlmAgent` with Gemini queries MCP servers (`arxiv-mcp`,
    `zotero-mcp`)\
3. Results are normalized into structured JSON and displayed as **paper
    cards** (title, authors, year, abstract, links)\
4. Users provide **likes/dislikes** → a built-in reranker uses Gemini
    embeddings to reorder results\
5. Users can:
    - Re-rank with feedback\
    - Generate a **gap analysis**\
    - Draft a **literature review**\
    - **Export liked papers to Zotero**

------------------------------------------------------------------------

## Working Code

- **Streamlit app:** `src/app.py`\
    (UI, ADK agent wiring, session handling, feedback UI, rerank/export
    controls)\
- **Reranking:** `src/rerank/rerank.py`\
    (Gemini embeddings with caching; feedback-based scoring)\
- **Tracing:** Opik integrated with ADK callbacks\
- **Tests:** `tests/` (`pytest`-based)

------------------------------------------------------------------------

## Repository Structure

    src/
    ├── app.py        # Streamlit app + agent pipeline
    └── rerank/
        └── rerank.py # Embedding + rerank utilities
    tests/            # Pytest tests
    notebooks/        # Experiments
    Makefile          # Convenience commands
    README.md

------------------------------------------------------------------------

## Setup

``` bash
git clone <repository-url>
cd ai-tinkerers-hackathon
uv sync --dev
cp -v .env.example .env
# Edit .env to set GEMINI_API_KEY, ZOTERO_USER_ID, ZOTERO_API_KEY
```

------------------------------------------------------------------------

## Gemini Integration

- Models:
  - Chat/retrieval: `gemini-2.5-flash-lite`\
  - Gap/Review: `gemini-2.5-pro`\
  - Embeddings: `gemini-embedding-001`\
- Tool calling:
  - ADK MCP toolset\
  - `arxiv-mcp-server` (search/download arXiv)\
  - `zotero-mcp` (Zotero collections/items)\
- Human feedback → rerank → traced via **Opik**

------------------------------------------------------------------------

## Features

- **Agent-driven retrieval** from arXiv and Zotero\
- **Paper cards** with titles, authors, year, abstract, and links\
- **Likes/Dislikes** with reranking (Gemini embeddings)\
- **Gap Analysis** and **Literature Review** generation (streaming
    responses)\
- **Export to Zotero** (liked papers)\
- **Opik tracing** for experiments

------------------------------------------------------------------------

## Limitations

- UI is **text-only** (no in-app PDF parsing)\
- Rerank quality depends on limited user feedback signals\
- Cloud deployment not included (local demo only)

------------------------------------------------------------------------

## Deployment Status

- **Local Streamlit app**\
- MCP servers launched via `uv tool run`\
- Zotero integration via **pyzotero**

------------------------------------------------------------------------

## Technologies

- Streamlit, Google ADK (`google.adk`), MCP (`arxiv-mcp-server`,
    `zotero-mcp`)\
- Gemini via `google.genai`\
- Opik (tracing), Python 3, `uv` (Astral)\
- Pytest, pre-commit

------------------------------------------------------------------------

## How to Run

Using the included Makefile:

``` bash
make install   # Install deps + setup zotero-mcp
make run       # Run app
```

Or directly:

``` bash
uv run streamlit run src/app.py
# then open http://localhost:8501
```

------------------------------------------------------------------------

## Tests

``` bash
uv run pytest
```

------------------------------------------------------------------------

## Key Commands

- **Start UI:**\
    `uv run streamlit run src/app.py`\
- **Tests:**\
    `uv run pytest`\
- **Clear session state:** in-app sidebar\
- **Local URL:**\
    `http://localhost:8501`

------------------------------------------------------------------------

✨ With Zotero export and streaming gap/review generation, this app now
delivers a complete **literature exploration workflow**: search → rerank
→ synthesize → save.
