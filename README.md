# AI Tinkerers Hackathon Project: Research Assistant

## Project Description
This project is a Streamlit-based research assistant that searches arXiv via an MCP tool, summarizes and organizes results, and lets users like/dislike papers to rerank recommendations. It demonstrates an end-to-end agentic workflow with Google's ADK, Gemini models, tool calling over MCP, and human-in-the-loop feedback that influences retrieval ranking.

## Technical Excellence (End-to-End Demo & Working Code)
### End-to-End Demo
1. User enters a research topic in the Streamlit UI  
2. An ADK `LlmAgent` using Gemini calls the arXiv MCP server (over stdio) to search/fetch papers  
3. Results are rendered as markdown "paper cards" with titles, authors, abstracts, and links  
4. Users provide likes/dislikes; a built-in reranker uses Gemini embeddings to reorder results based on user feedback  

### Working Code
- Core app logic: `src/app.py` (Streamlit UI, ADK agent wiring, session handling, human feedback UI, rerank controls)
- Prompting: `src/prompt.py` (task-specific system prompt with clarification protocol and output format)
- Reranking: `src/rerank/rerank.py` (Gemini embeddings with local caching; maps feedback into a rerank score)
- Agent sample: `src/main.py` (minimal ADK/MCP agent example)
- Tracing: Opik tracer integrated with ADK callbacks
- Tests: Run with `uv run pytest`

## Solution Architecture & Documentation
### Repository Structure

src/
├── app.py # Streamlit app and ADK runner glue
├── prompt.py # Assistant instructions
├── main.py # Minimal ADK/MCP agent example
└── rerank/
└── rerank.py # Embedding + rerank utilities
tests/ # Pytest tests
notebooks/ # Experiments
README.md # Setup and usage

### Setup
```bash
git clone <repository-url>
cd ai-tinkerers-hackathon
uv sync --dev
cp -v .env.example .env
# Edit .env to set GEMINI_API_KEY
```

## Gemini Integration
- Models:
  - Chat: `gemini-2.5-flash-lite` (in `src/app.py`)
  - Embeddings: `gemini-embedding-001` (in `src/rerank/rerank.py`)
- Tool calling: ADK MCP toolset to `arxiv-mcp-server` over stdio (`uv tool run arxiv-mcp-server`), tools include `search_papers`, `download_paper`, `list_papers`, `read_paper`
- Multimodal: Text-only in current UI
- Chaining: User query → ADK agent with MCP tools → formatted results → human feedback → embedding-based rerank; traced via Opik
- Evaluation: Opik tracing + pytest; feedback-driven rerank as a practical signal

## Impact & Innovation
- Impact: Accelerates literature review with agentic search and feedback-driven ranking
- Innovation: ADK+MCP integration plus human-in-the-loop reranking with Gemini embeddings, delivered in a clean, reproducible Streamlit experience

## Features
- Agent-driven arXiv search using MCP
- Clarification protocol for missing constraints (year range, count, survey toggle)
- Paper cards with titles/authors/abstract/links
- Like/Dislike feedback; save/load preferences
- Rerank using Gemini embeddings guided by feedback
- Opik tracing; example queries and quick-start sidebar

## Experimental (Not in Demo)
- Gap Analysis Agent: drafts research gap analyses across returned papers in a topic area (early prototype; still stabilizing outputs)
- Literature Review Agent: composes structured literature reviews (motivation, methods, findings, limitations) from curated sets (in-progress)
- Enhanced Prompting: a more complex prompt that elicits additional clarifying questions beyond defaults for higher-precision retrieval (partially implemented)

## Limitations
- Text-only interaction; no in-UI PDF parsing
- Simple embedding-based rerank; quality depends on limited feedback signals
- Local demo focus; no managed cloud deployment
- Requires `GEMINI_API_KEY` for embeddings

## Deployment Status
- Local Streamlit app; MCP server launched via uv tool over stdio
- Intended for local demo and judging; cloud deployment not included

## Technologies
- Streamlit, Google ADK (`google.adk`), MCP (`arxiv-mcp-server`)
- Gemini via `google.genai` (`gemini-2.5-flash-lite`, `gemini-embedding-001`)
- Opik (tracing), Python 3, `uv` (Astral), Pytest, pre-commit

## How to Run
```bash
uv run streamlit run src/app.py
# then open http://localhost:8501
```

## Tests
```bash
uv run pytest
```

## Key Commands
- Start UI: `uv run streamlit run src/app.py`
- Tests: `uv run pytest`
- Local URL: `http://localhost:8501`
