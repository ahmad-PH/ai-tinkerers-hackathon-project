# app.py
import asyncio
import hashlib
import json
import logging
import os
import re
from time import sleep

import streamlit as st
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types
from mcp import StdioServerParameters
from opik import configure
from opik.integrations.adk import OpikTracer
from pydantic import BaseModel, Field

# Optional pyzotero for better collection picker
try:
    from pyzotero import zotero as pyzotero_mod

    HAVE_PYZOTERO = True
except Exception:
    HAVE_PYZOTERO = False

# Your reranker
from src.rerank.rerank import Paper, rerank

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
configure()
load_dotenv()

for noisy in ("mcp", "google.adk.tools.mcp_tool", "google.adk", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "research_assistant_app"
USER_ID = "local_user"
SESSION_ID = "default_session"

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")


class RetrievalSet(BaseModel):
    papers: list[Paper] = Field(default_factory=list)
    notes: str = ""


# -----------------------------------------------------------------------------
# Helpers: env & MCP toolsets
# -----------------------------------------------------------------------------


def zotero_env_ok() -> bool:
    return bool(os.getenv("ZOTERO_USER_ID") and os.getenv("ZOTERO_API_KEY"))


def get_zotero_client():
    if not HAVE_PYZOTERO or not zotero_env_ok():
        return None
    lib_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    user_id = os.getenv("ZOTERO_USER_ID")
    api_key = os.getenv("ZOTERO_API_KEY")
    try:
        return pyzotero_mod.Zotero(user_id, lib_type, api_key)
    except Exception:
        return None


def list_zotero_collections() -> list[tuple[str, str]]:
    z = get_zotero_client()
    if not z:
        return []
    try:
        cols = z.collections()
        out = []
        for c in cols:
            data = c.get("data", {})
            out.append((data.get("name", "Untitled"), data.get("key", "")))
        return out
    except Exception:
        return []


# ---- Zotero export helpers ---------------------------------------------------
def _paper_to_zotero_item(p: Paper, collection_key: str | None = None) -> dict:
    """Map our Paper -> Zotero 'journalArticle' (or 'preprint' if no year/venue)."""
    # Authors to Zotero creator schema
    creators = []
    for a in p.authors or []:
        # naive split "Last, First" or "First Last"
        if "," in a:
            last, first = [s.strip() for s in a.split(",", 1)]
        else:
            parts = a.split()
            first = " ".join(parts[:-1]).strip() if len(parts) > 1 else ""
            last = parts[-1] if parts else ""
        creators.append({"creatorType": "author", "firstName": first, "lastName": last})

    item = {
        "itemType": "journalArticle",  # most arXiv works later get proper metadata
        "title": p.title or "",
        "abstractNote": p.abstract or "",
        "creators": creators,
        "date": str(p.year) if isinstance(p.year, int) and p.year > 0 else "",
        "DOI": (p.doi or "").strip(),
        "url": p.url or "",
        "libraryCatalog": "ADK Research Assistant",
        "tags": [{"tag": "adk-import"}],
    }
    # If no DOI/year, tag as preprint-like for clarity
    if not item["DOI"] and not item["date"]:
        item["extra"] = "arXiv import (metadata incomplete)"
    if collection_key:
        item["collections"] = [collection_key]
    return item


def _find_existing_item(z, p: Paper):
    """Try to find an existing Zotero item by DOI (best) or title+year (fallback)."""
    try:
        if p.doi:
            # DOI search (works well)
            found = z.items(q=p.doi, qmode="everything", itemType="-attachment")
            if found:
                return found[0]
        # Fallback: title (quoted) and year filter
        query = f'"{p.title}"' if p.title else ""
        if query:
            found = z.items(q=query, qmode="titleCreatorYear", itemType="-attachment")
            if found and isinstance(p.year, int) and p.year > 0:
                for it in found:
                    data = it.get("data", {})
                    if (data.get("date", "") or "").startswith(str(p.year)):
                        return it
            elif found:
                return found[0]
    except Exception:
        pass
    return None


def export_to_zotero(papers: list[Paper], collection_key: str | None = None, dry_run: bool = False) -> dict:
    """
    Create (or skip existing) items in Zotero using pyzotero.
    Returns {created:[], skipped:[], failed:[(paper, error_str)]}
    """
    res = {"created": [], "skipped": [], "failed": []}
    z = get_zotero_client()
    if not z:
        raise RuntimeError("Zotero client not available. Check ZOTERO_USER_ID, ZOTERO_API_KEY, and pyzotero install.")

    # Prepare items with de-duplication check
    to_create = []
    for p in papers:
        existing = _find_existing_item(z, p)
        if existing:
            res["skipped"].append({"title": p.title, "reason": "exists", "key": existing.get("key")})
            continue
        item = _paper_to_zotero_item(p, collection_key=collection_key or None)
        to_create.append(item)

    if dry_run or not to_create:
        return res

    # Create in small batches with retry for server hiccups
    BATCH = 10
    for i in range(0, len(to_create), BATCH):
        batch = to_create[i : i + BATCH]
        # two tries per batch
        for attempt in (1, 2):
            try:
                created = z.create_items(batch)
                # pyzotero returns a dict keyed by successful keys or a list; normalize
                if isinstance(created, dict):
                    # successes are under 'successful'
                    for k, v in (created.get("successful") or {}).items():
                        res["created"].append({"key": k, "title": v.get("data", {}).get("title", "")})
                    # show failures per-item if any
                    for _, v in (created.get("failed") or {}).items():
                        title = v.get("data", {}).get("title", "")
                        err = v.get("error", "unknown error")
                        res["failed"].append((title, f"create failed: {err}"))
                else:
                    # fallback if API returns list
                    for v in created or []:
                        res["created"].append({"key": v.get("key"), "title": v.get("data", {}).get("title", "")})
                break  # batch ok
            except Exception:
                if attempt == 1:
                    sleep(0.8)  # brief backoff and retry once
                else:
                    # On persistent 5xx: fall back to per-item create to salvage what we can
                    for it in batch:
                        try:
                            v = z.create_items([it])
                            if isinstance(v, dict) and v.get("successful"):
                                for k2, v2 in v["successful"].items():
                                    res["created"].append({"key": k2, "title": v2.get("data", {}).get("title", "")})
                            elif isinstance(v, list) and v:
                                res["created"].append(
                                    {"key": v[0].get("key"), "title": v[0].get("data", {}).get("title", "")}
                                )
                            else:
                                res["failed"].append((it.get("title", ""), "unknown create response"))
                        except Exception as ei:
                            res["failed"].append((it.get("title", ""), str(ei)))
    return res


def _split_author(name: str) -> dict:
    """Very light name splitter for Zotero creators."""
    name = (name or "").strip()
    if not name:
        return {"creatorType": "author", "firstName": "", "lastName": ""}
    # If name contains comma, assume "Last, First"
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return {"creatorType": "author", "firstName": first, "lastName": last}
    # Else assume last token is last name
    parts = name.split()
    if len(parts) == 1:
        return {"creatorType": "author", "firstName": "", "lastName": parts[0]}
    return {"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]}


def mcp_toolset(command: str, args: list[str], env: dict | None = None, timeout: float = 60.0):
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command=command, args=args, env=env or {}),
            timeout=timeout,
        )
    )


def arxiv_mcp_toolset():
    return mcp_toolset("uv", ["tool", "run", "arxiv-mcp-server"])


def zotero_mcp_toolset():
    # Provide both expected env names to be safe across repos
    env = {
        "ZOTERO_LIBRARY_ID": os.getenv("ZOTERO_USER_ID", ""),
        "ZOTERO_USER_ID": os.getenv("ZOTERO_USER_ID", ""),
        "ZOTERO_API_KEY": os.getenv("ZOTERO_API_KEY", ""),
        "ZOTERO_LIBRARY_TYPE": os.getenv("ZOTERO_LIBRARY_TYPE", "user"),
        "ZOTERO_LOCAL": os.getenv("ZOTERO_LOCAL", "false"),
    }
    # Common CLI name in your repo is "zotero-mcp"; adjust if needed
    return mcp_toolset("uvx", ["zotero-mcp"], env=env)


def ping_badge(ok: bool) -> str:
    return "✅" if ok else "❌"


# -----------------------------------------------------------------------------
# Streamlit caches: runner + tracer
# -----------------------------------------------------------------------------
@st.cache_resource
def get_asyncio_runner() -> asyncio.Runner:
    # One event loop for the lifetime of the Streamlit session
    return asyncio.Runner()


@st.cache_resource
def create_tracer():
    return OpikTracer(
        name="research-assistant-tracer",
        tags=["streamlit", "arxiv", "zotero", "mcp", "adk"],
        metadata={"environment": "development", "framework": "google-adk", "feature": "research-assistant"},
        project_name="adk-research-assistant",
    )


# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------


def create_retriever_agent(tracer, arxiv_ts, zotero_ts):
    instruction = """You are a retrieval-focused research assistant.

# OUTPUT POLICY — IMPORTANT
Return *only* a single JSON object inside one fenced code block. No prose before/after.
If a field is unknown, use "" (empty string) or 0 for year. Follow the exact keys in the schema.

Goal: return MANY relevant papers as structured content for downstream use. You may call tools multiple times (ReAct-style).
No clustering. Be explicit and iterative up to CONFIG.search_iterations.

Behavior:
1) If CONFIG.use_zotero=true:
   a) Use Zotero MCP to list seed items (collection key OR tag; respect CONFIG.zotero_recent).
   b) For each seed item, try:
      (i) Read attached PDF to extract title + abstract (or first paragraph).
      (ii) Else fetch abstract from arXiv by title/DOI.
      (iii) Else fallback to Zotero metadata.
   c) Produce a 1–2 sentence theme summary in "notes".

2) Expansion (iterative up to CONFIG.search_iterations):
   - Formulate focused arXiv queries; enforce CONFIG.cutoff.
   - On each iteration: search, add unique candidates, deduplicate, refine queries.
   - Stop at ~CONFIG.top_k or iteration cap.

Schema to emit (keys must match exactly):
{
  "papers": [
    {
      "source": "arxiv or zotero",
      "id": "string",
      "title": "string",
      "authors": ["..."],
      "year": 2024,             // -1 if unknown
      "doi": "string",
      "url": "string",
      "abstract": "string",
      "why_relevant": "string"
    }
  ],
  "notes": "string"
}
"""
    return LlmAgent(
        model="gemini-2.5-flash-lite",
        name="retriever_agent",
        instruction=instruction,
        tools=[arxiv_ts, zotero_ts],
        output_schema=RetrievalSet,
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
        before_tool_callback=tracer.before_tool_callback,
        after_tool_callback=tracer.after_tool_callback,
    )


def create_structurer_agent(tracer):
    # We’ll still parse JSON locally, but we tell the model to strictly emit the schema
    return LlmAgent(
        model="gemini-2.5-flash",
        name="structurer_agent",
        instruction=(
            "Convert the user's previous text into a strict RetrievalSet JSON (no commentary).\n"
            "If fields are missing, fill with empty strings or nulls appropriately. "
            "Keep abstracts as-is (no truncation). Output only the JSON."
        ),
        # No tools
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
    )


def create_gap_agent(tracer):
    instruction = """You are a gap-analysis agent.

Given JSON with a list of retrieved papers (title, abstract, year, doi/url) and the original idea/context,
produce a concise gap analysis (no clustering). Focus on:
- patterns/themes,
- inconsistencies or contradictions,
- under-explored populations/settings/methods/data,
- measurement or reproducibility limitations,
- concrete opportunities (testable directions).

Output markdown with sections:
- **Idea recap** (1–2 sentences)
- **Patterns observed** (3–5 bullets with inline refs by title or DOI)
- **Inconsistencies / limitations** (3–5 bullets with refs)
- **Under-explored opportunities** (4–8 bullets; each bullet cites 1–2 supporting papers)
- **Suggested next-read list** (10–15 items as Title (Year) — link)
Keep it tight and grounded to the provided papers. If the list is small, say so and be conservative."""
    return LlmAgent(
        model="gemini-2.5-pro",
        name="gap_agent",
        instruction=instruction,
        tools=[],
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
    )


def create_review_agent(tracer):
    instruction = """You are a literature-review writing agent.

Given JSON with retrieved papers and the research idea/context, draft a literature review section suitable for a paper.
No clustering. Structure:

**Background & scope**: 1 paragraph tying the idea to major strands
**Synthesis of prior work**: 2–4 short paragraphs organized by themes/methods,
  weaving in citations by Title (Year) or DOI (avoid numeric citations).
**Methods & data seen**: 1 short paragraph (common study designs/datasets)
**Gaps motivating new work**: 1 paragraph (reference the earlier synthesis)

Keep it 400–700 words. Maintain academic tone but crisp. Ground claims with references to specific papers from the provided list."""
    return LlmAgent(
        model="gemini-2.5-pro",
        name="review_agent",
        instruction=instruction,
        tools=[],
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
    )


# -----------------------------------------------------------------------------
# Agent + runner factory
# -----------------------------------------------------------------------------
@st.cache_resource
def create_agents_and_runners():
    tracer = create_tracer()
    arxiv_ts = arxiv_mcp_toolset()
    zotero_ts = zotero_mcp_toolset()

    retriever = create_retriever_agent(tracer, arxiv_ts, zotero_ts)
    structurer = create_structurer_agent(tracer)
    gap = create_gap_agent(tracer)
    review = create_review_agent(tracer)

    sess = InMemorySessionService()
    run_retriever = Runner(agent=retriever, app_name=APP_NAME, session_service=sess)
    run_structurer = Runner(agent=structurer, app_name=APP_NAME, session_service=sess)
    run_gap = Runner(agent=gap, app_name=APP_NAME, session_service=sess)
    run_review = Runner(agent=review, app_name=APP_NAME, session_service=sess)

    asyncio.run(sess.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID + "_retriever"))
    asyncio.run(sess.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID + "_structurer"))
    asyncio.run(sess.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID + "_gap"))
    asyncio.run(sess.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID + "_review"))

    return (retriever, structurer, gap, review, run_retriever, run_structurer, run_gap, run_review)


# -----------------------------------------------------------------------------
# Async runner helper
# -----------------------------------------------------------------------------


async def _run_turn_async(runner, user_id, session_id, content):
    def _extract_texts_from_parts(parts) -> list[str]:
        texts: list[str] = []

        for p in parts or []:
            # 1) Normal model text parts
            txt = getattr(p, "text", None)
            if isinstance(txt, str) and txt.strip():
                texts.append(txt)
                continue

            # 2) Tool / function responses (ADK wraps these as Pydantic objects)
            fr = getattr(p, "function_response", None)
            if fr is None:
                # Some builds embed the tool result directly in the part
                # (keep this as a no-op if not present)
                continue

            # Normalize: prefer attribute access, then dict fallback
            resp = getattr(fr, "response", None)
            if resp is None and isinstance(fr, dict):
                resp = fr.get("response")

            # ADK sometimes exposes result either under response.result or directly as fr.result
            result = getattr(resp, "result", None) if resp is not None else None
            if result is None:
                result = getattr(fr, "result", None)
            if result is None and isinstance(resp, dict):
                result = resp.get("result")

            # Now drill into the content list (attribute or dict)
            content_list = None
            if result is not None:
                content_list = getattr(result, "content", None)
                if content_list is None and isinstance(result, dict):
                    content_list = result.get("content")

            # Some tools return a simple string instead of a content list
            if content_list is None and result is not None:
                alt_text = getattr(result, "text", None)
                if alt_text is None and isinstance(result, dict):
                    alt_text = result.get("text")
                if isinstance(alt_text, str) and alt_text.strip():
                    texts.append(alt_text)
                    continue

            # Typical MCP shape: list of items with type/text
            for c in content_list or []:
                if isinstance(c, dict):
                    ctype = c.get("type")
                    ctext = c.get("text")
                    if ctype == "text" and isinstance(ctext, str) and ctext.strip():
                        texts.append(ctext)
                else:
                    ctype = getattr(c, "type", None)
                    ctext = getattr(c, "text", None)
                    if ctype == "text" and isinstance(ctext, str) and ctext.strip():
                        texts.append(ctext)

        return texts

    final_texts: list[str] = []
    last_tool_texts: list[str] = []

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        ec = getattr(event, "content", None)
        if ec is not None and getattr(ec, "parts", None):
            # Plain model text
            for p in ec.parts:
                pt = getattr(p, "text", None)
                if isinstance(pt, str) and pt.strip():
                    final_texts.append(pt)
            # Tool text (important)
            ttexts = _extract_texts_from_parts(ec.parts)
            if ttexts:
                last_tool_texts = ttexts

        if event.is_final_response():
            break

    txt = "".join(final_texts).strip()
    if not txt and last_tool_texts:
        txt = "\n\n".join(last_tool_texts).strip()
    return txt or ""


# -----------------------------------------------------------------------------
# JSON extraction + structuring
# -----------------------------------------------------------------------------
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def extract_json_block(text: str) -> dict | None:
    m = JSON_BLOCK_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _year_from_any(v) -> int:
    """Extract a 4-digit year or return -1 if unknown."""
    if isinstance(v, int):
        return v
    if not v:
        return -1
    # try ISO date first
    try:
        return int(str(v)[:4])
    except Exception:
        pass
    m = re.search(r"\b(19|20)\d{2}\b", str(v))
    return int(m.group(0)) if m else -1


def to_retrieval_set(data: dict) -> RetrievalSet:
    """
    Normalize heterogeneous tool output into the strict RetrievalSet schema.
    - If it's already compliant, pass through.
    - If it's arXiv MCP-style, map fields and fill defaults.
    """
    if not isinstance(data, dict):
        data = {}  # avoid NoneType splats

    raw_papers = data.get("papers") or []

    # Quick path: looks already compliant (has "source" on first item)
    if raw_papers and isinstance(raw_papers[0], dict) and "source" in raw_papers[0]:
        return RetrievalSet(**data)

    # Normalize arXiv-shape to your schema
    norm_papers = []
    for p in raw_papers:
        if not isinstance(p, dict):
            continue
        arxiv_id = p.get("id") or ""
        title = p.get("title") or ""
        authors = p.get("authors") or []
        abstract = p.get("abstract") or ""
        published = p.get("published") or p.get("date") or ""
        url = p.get("url") or f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
        doi = p.get("doi") or ""
        why_rel = p.get("why_relevant") or ""  # tools usually won't provide this

        # Ensure authors is a list[str]
        if isinstance(authors, str):
            authors = [a.strip() for a in re.split(r",| and ", authors) if a.strip()]
        elif not isinstance(authors, list):
            authors = []

        norm_papers.append(
            {
                "source": "arxiv",  # arXiv MCP doesn't set this
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "year": _year_from_any(published),  # int (or -1)
                "doi": doi,
                "url": url,
                "abstract": abstract,
                "why_relevant": why_rel,
            }
        )

    normalized = {
        "papers": norm_papers,
        "notes": data.get("notes") or "",
    }
    return RetrievalSet(**normalized)


def _parse_zotero_markdown(md: str) -> list[dict]:
    """Turn Zotero MCP markdown listing into minimal Paper dicts."""
    titles = []
    for m in re.finditer(r"^##\s*\d+\.\s*(.+)$", md, flags=re.MULTILINE):
        t = m.group(1).strip()
        # skip obvious attachments/notes
        if t.lower().endswith(".pdf"):
            continue
        if t.lower().startswith("untitled"):
            continue
        titles.append(t)

    papers = []
    for t in titles:
        papers.append(
            {
                "source": "zotero",
                "id": "",
                "title": t,
                "authors": [],
                "year": -1,
                "doi": "",
                "url": "",
                "abstract": "",
                "why_relevant": "",
            }
        )
    return papers


# -----------------------------------------------------------------------------
# UI: sidebar
# -----------------------------------------------------------------------------
st.title("📚 Research Assistant")
st.caption("Google ADK • arXiv MCP • Zotero MCP • Opik")

with st.sidebar:
    st.header("🔧 Mode & Sources")
    mode = st.radio("Start from", ["Idea", "Zotero"], index=0, horizontal=True)
    use_zotero = st.checkbox("Use Zotero seeds", value=(mode == "Zotero"))

    selected_collection_key = ""
    selected_collection_name = ""
    z_tag = ""
    z_recent = 20

    if use_zotero:
        if zotero_env_ok():
            st.success(f"Zotero creds found {ping_badge(True)}")
            if HAVE_PYZOTERO:
                cols = list_zotero_collections()
                if cols:
                    selected = st.selectbox("Zotero collection", options=cols, format_func=lambda x: x[0], index=0)
                    selected_collection_name, selected_collection_key = selected
                else:
                    st.warning("No collections found via pyzotero. You can still specify a tag below.")
            else:
                st.warning("pyzotero not installed; using text inputs instead.")
        else:
            st.error(f"Zotero not configured {ping_badge(False)} — set ZOTERO_USER_ID and ZOTERO_API_KEY")

        z_tag = st.text_input("Optional Zotero tag", placeholder="e.g. diffusion")
        z_recent = st.number_input("Recent N items", min_value=0, max_value=100, value=20, step=5)

    st.markdown("### Retrieval controls")
    top_k = st.slider("Top-k papers", min_value=3, max_value=50, value=10, step=5)
    cutoff = st.text_input("Date cutoff (YYYY-MM-DD, optional)", placeholder="e.g. 2023-12-31")
    search_iterations = st.slider("Search iterations (ReAct depth)", 1, 3, value=2, step=1)

    st.markdown("---")
    st.caption(f"arXiv MCP: {ping_badge(True)} (assumes CLI available)")
    st.caption(f"Zotero MCP: {ping_badge(zotero_env_ok())}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.pop("messages", None)
        st.session_state.pop("paper_feedback", None)
        st.session_state.pop("last_ranked_papers", None)
        st.rerun()

# -----------------------------------------------------------------------------
# Session init
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "paper_feedback" not in st.session_state:
    st.session_state.paper_feedback = {}  # key -> {'vote': 'like'|'dislike'|None}
if "user_id" not in st.session_state:
    st.session_state.user_id = USER_ID
if "session_id" not in st.session_state:
    st.session_state.session_id = SESSION_ID

# Multi-agent runners (create once)
if "runner_retriever" not in st.session_state:
    with st.spinner("Initializing multi-agent pipeline…"):
        (
            st.session_state.agent_retriever,
            st.session_state.agent_structurer,
            st.session_state.agent_gap,
            st.session_state.agent_review,
            st.session_state.runner_retriever,
            st.session_state.runner_structurer,
            st.session_state.runner_gap,
            st.session_state.runner_review,
        ) = create_agents_and_runners()

# -----------------------------------------------------------------------------
# Render history
# -----------------------------------------------------------------------------


def normalize_title(s: str) -> str:
    s = s or "Untitled"
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s).strip()
    s = re.sub(r"^(?:\d+\.\s+|[-*+]\s+|#+\s+)", "", s).strip("*_` ").strip()
    s = re.sub(r"\s+", " ", s)
    return s or "Untitled"


def _paper_key(title: str, url: str | None) -> str:
    return f"{normalize_title(title)}|{url or ''}"


def _paper_hash_key(title: str, url: str | None) -> str:
    return hashlib.sha1(_paper_key(title, url).encode("utf-8")).hexdigest()[:10]


def render_cards(papers: list[Paper], namespace: str):
    for i, p in enumerate(papers, start=1):
        # Stable keys: one for state (by title+url), one hashed for Streamlit widgets
        key_struct = _paper_key(p.title, p.url)
        key_hash = _paper_hash_key(p.title, p.url)
        vote = st.session_state.paper_feedback.get(key_struct, {}).get("vote")

        with st.container(border=True):
            # Title
            st.markdown(f"### {i}. {p.title}")

            # Authors (+ optional year)
            authors = ", ".join(p.authors) if isinstance(p.authors, list) and p.authors else ""
            year_str = f" ({p.year})" if isinstance(p.year, int) and p.year > 0 else ""
            if authors or year_str:
                st.caption(f"{authors}{year_str}")

            # URL (or DOI fallback)
            if p.url:
                st.markdown(f"[Open paper]({p.url})")
            elif p.doi:
                st.markdown(f"[Open via DOI](https://doi.org/{p.doi})")

            # Abstract (collapsible)
            if p.abstract:
                with st.expander("Abstract", expanded=False):
                    st.write(p.abstract)

            # Optional short rationale
            if getattr(p, "why_relevant", ""):
                st.caption(f"Why relevant: {p.why_relevant}")

            # Feedback buttons
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("👍 Like", key=f"like_{namespace}_{key_hash}", disabled=(vote == "like")):
                    st.session_state.paper_feedback[key_struct] = {"vote": "like"}
                    st.rerun()
            with c2:
                if st.button("👎 Dislike", key=f"dislike_{namespace}_{key_hash}", disabled=(vote == "dislike")):
                    st.session_state.paper_feedback[key_struct] = {"vote": "dislike"}
                    st.rerun()


for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict) and msg["content"].get("papers"):
            render_cards(
                [Paper(**p) if isinstance(p, dict) else p for p in msg["content"]["papers"]],
                namespace=f"hist_{idx}",  # unique per message
            )
        else:
            st.markdown(msg["content"] if isinstance(msg["content"], str) else str(msg["content"]))


# -----------------------------------------------------------------------------
# Orchestrator: one turn
# -----------------------------------------------------------------------------


def run_pipeline(user_prompt: str, cfg: dict) -> RetrievalSet | None:
    loop_runner = get_asyncio_runner()

    # 1) RETRIEVE
    retriever_content = types.Content(
        role="user",
        parts=[
            types.Part(
                text=("CONFIG:\n" + "\n".join(f"{k}={v}" for k, v in cfg.items()) + "\n\nUSER:\n" + user_prompt)
            )
        ],
    )
    raw_text = loop_runner.run(
        _run_turn_async(
            st.session_state.runner_retriever,
            st.session_state.user_id,
            st.session_state.session_id + "_retriever",
            retriever_content,
        )
    )

    # Save raw output for debugging
    st.session_state.last_raw_retriever = raw_text

    # Try several ways to get JSON
    json_obj = extract_json_block(raw_text)
    if json_obj is None:
        try:
            json_obj = json.loads(raw_text)
        except Exception:
            json_obj = None

    # 🔁 Zotero-markdown fallback (safe for non-Zotero: condition won’t match)
    if json_obj is None:
        rt = (raw_text or "").lstrip()
        looks_like_zotero_md = (
            rt.startswith("# Items in Collection")
            or ("**Item Key:**" in rt and rt.startswith("#"))
            or re.search(r"^##\s*\d+\.\s+.+", rt, flags=re.MULTILINE)
        )
        if looks_like_zotero_md:
            parsed = _parse_zotero_markdown(rt)
            if parsed:
                json_obj = {"papers": parsed, "notes": "Parsed from Zotero markdown list."}

    # If still None, try the structurer repair once
    if json_obj is None:
        struct_input = f"Please convert the following into valid RetrievalSet JSON only:\n\n{raw_text}"
        struct_content = types.Content(role="user", parts=[types.Part(text=struct_input)])
        structured_text = loop_runner.run(
            _run_turn_async(
                st.session_state.runner_structurer,
                st.session_state.user_id,
                st.session_state.session_id + "_structurer",
                struct_content,
            )
        )
        # try parsing the repaired text
        json_obj = extract_json_block(structured_text)
        if json_obj is None:
            try:
                json_obj = json.loads(structured_text)
            except Exception:
                json_obj = None

    # If still None, bail gracefully with an empty set (and surface raw output)
    if json_obj is None:
        st.warning("Retriever produced no valid JSON. See the raw output under the debug expander.")
        return RetrievalSet(papers=[], notes="(empty)")

    # Finally coerce whatever we got into a RetrievalSet (this is now null-safe)
    retrieval = to_retrieval_set(json_obj)

    # Enforce top_k
    if cfg.get("top_k"):
        retrieval.papers = retrieval.papers[: int(cfg["top_k"])]

    return retrieval


# -----------------------------------------------------------------------------
# Chat input
# -----------------------------------------------------------------------------

prompt = st.chat_input("Type a research idea, topic, or a Zotero instruction…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_top_k = top_k
    cfg = {
        "use_zotero": str(bool(use_zotero)).lower(),
        "zotero_collection_key": selected_collection_key or "",
        "zotero_collection_name": selected_collection_name or "",
        "zotero_tag": z_tag if use_zotero else "",
        "zotero_recent": int(z_recent) if use_zotero else 0,
        "top_k": top_k,
        "cutoff": cutoff or "",
        "search_iterations": search_iterations,
    }

    with st.chat_message("assistant"):
        st.caption(
            f"Run config — Zotero: {use_zotero} • "
            f"Collection: {selected_collection_name or '—'} • "
            f"Key: {selected_collection_key or '—'} • Tag: {z_tag or '—'} • "
            f"Top-k: {top_k} • Iterations: {search_iterations} • Cutoff: {cutoff or '—'}"
        )
        placeholder = st.empty()
        with st.spinner("Retrieving…"):
            retrieval = run_pipeline(prompt, cfg)

        if retrieval is None or not retrieval.papers:
            placeholder.markdown("No structured results found.")
        else:
            # First-pass rerank (no feedback)
            ranked = rerank(
                query=prompt,
                papers=retrieval.papers,
                user_feedbacks=None,
            )

            st.session_state.last_ranked_papers = [p if isinstance(p, Paper) else Paper(**p) for p in ranked]
            st.session_state.show_synthesis_buttons = True
            # Render as cards & store as an assistant turn (structured)
            render_cards(st.session_state.last_ranked_papers, namespace="live")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {"papers": [p.model_dump() for p in st.session_state.last_ranked_papers]},
                }
            )


def _get_selected_for_synthesis(top_k_default: int = None) -> list[Paper]:
    liked = []
    for p in st.session_state.get("last_ranked_papers", []):
        k = _paper_key(p.title, p.url)  # <— match render_cards/_paper_key
        vote = st.session_state.paper_feedback.get(k, {}).get("vote")
        if vote == "like":
            liked.append(p)
    if liked:
        return liked
    k = st.session_state.get("last_top_k") or top_k_default or len(st.session_state.get("last_ranked_papers", []))
    return st.session_state.get("last_ranked_papers", [])[:k]


# -----------------------------------------------------------------------------
# Rerank button + synthesis buttons
# -----------------------------------------------------------------------------

if st.session_state.get("last_ranked_papers") and st.session_state.get("show_synthesis_buttons", False):
    with st.container():
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

        with c1:
            if st.button("🔀 Rerank with feedback", key="btn_rerank_after"):
                feedback_vec = []
                for p in st.session_state.last_ranked_papers:
                    k = _paper_key(p.title, p.url)
                    vote = st.session_state.paper_feedback.get(k, {}).get("vote")
                    feedback_vec.append(1 if vote == "like" else (-1 if vote == "dislike" else 0))

                user_query = next(
                    (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), ""
                )
                try:
                    new_rank = rerank(
                        query=user_query, papers=st.session_state.last_ranked_papers, user_feedbacks=feedback_vec
                    )
                except Exception:
                    new_rank = rerank(
                        query=user_query,
                        papers=[p.model_dump() for p in st.session_state.last_ranked_papers],
                        user_feedbacks=feedback_vec,
                    )
                    new_rank = [Paper(**(p if isinstance(p, dict) else p.__dict__)) for p in new_rank]

                with st.chat_message("assistant"):
                    st.markdown("### Reranked (with feedback)")
                    render_cards(new_rank, namespace="reranked")
                st.session_state.messages.append(
                    {"role": "assistant", "content": {"papers": [p.model_dump() for p in new_rank]}}
                )
                st.session_state.last_ranked_papers = new_rank
                st.session_state.show_synthesis_buttons = True
                st.rerun()

        with c2:
            if st.button("🧭 Generate Gap Analysis", key="btn_gap_after"):
                selected = _get_selected_for_synthesis()
                if not selected:
                    st.warning("No papers to analyze yet.")
                else:
                    loop_runner = get_asyncio_runner()
                    user_query = next(
                        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), ""
                    )
                    payload = RetrievalSet(papers=selected, notes="ranked subset").model_dump()
                    gap_input = f"CONTEXT:\n{user_query}\n\nRETRIEVED_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                    gap_content = types.Content(role="user", parts=[types.Part(text=gap_input)])
                    with st.chat_message("assistant"):
                        with st.spinner("Generating gap analysis…"):
                            gap_md = loop_runner.run(
                                _run_turn_async(
                                    st.session_state.runner_gap,
                                    st.session_state.user_id,
                                    st.session_state.session_id + "_gap",
                                    gap_content,
                                )
                            )
                            st.markdown(gap_md or "_No output_")
                            st.session_state.messages.append({"role": "assistant", "content": gap_md or "_No output_"})

        with c3:
            if st.button("📝 Generate Literature Review", key="btn_review_after"):
                selected = _get_selected_for_synthesis()
                if not selected:
                    st.warning("No papers to review yet.")
                else:
                    loop_runner = get_asyncio_runner()
                    user_query = next(
                        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), ""
                    )
                    payload = RetrievalSet(papers=selected, notes="selected for review").model_dump()
                    review_input = (
                        f"CONTEXT:\n{user_query}\n\nRETRIEVED_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                    )
                    review_content = types.Content(role="user", parts=[types.Part(text=review_input)])
                    with st.chat_message("assistant"):
                        with st.spinner("Drafting literature review…"):
                            review_md = loop_runner.run(
                                _run_turn_async(
                                    st.session_state.runner_review,
                                    st.session_state.user_id,
                                    st.session_state.session_id + "_review",
                                    review_content,
                                )
                            )
                            st.markdown(review_md or "_No output_")
                            st.session_state.messages.append(
                                {"role": "assistant", "content": review_md or "_No output_"}
                            )

        # NEW: Export to Zotero
        with c4:
            if st.button("📥 Export liked → Zotero", key="btn_export_zotero"):
                if not HAVE_PYZOTERO or not zotero_env_ok():
                    st.error("Zotero export unavailable. Install `pyzotero` and set ZOTERO_USER_ID / ZOTERO_API_KEY.")
                else:
                    selected = _get_selected_for_synthesis()
                    if not selected:
                        st.warning("No papers selected to export.")
                    else:
                        try:
                            result = export_to_zotero(
                                papers=selected, collection_key=(selected_collection_key or None)
                            )
                            created_n = len(result["created"])
                            skipped_n = len(result["skipped"])
                            failed_n = len(result["failed"])
                            st.success(
                                f"Zotero export complete — created: {created_n}, skipped (existing): {skipped_n}, failed: {failed_n}"
                            )
                            if failed_n:
                                with st.expander("Show export failures"):
                                    for title, err in result["failed"]:
                                        st.write(f"• **{title}** — {err}")
                        except Exception as e:
                            st.error(f"Export failed: {e}")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit • Google ADK • arXiv/Zotero MCP • Opik")
