# app.py
import asyncio
import logging
import os

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

# Optional: pyzotero for listing collections nicely
try:
    from pyzotero import zotero as pyzotero_mod  # noqa: F401

    HAVE_PYZOTERO = True
except Exception:
    HAVE_PYZOTERO = False

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
configure()
load_dotenv()

for noisy in ("mcp", "google.adk.tools.mcp_tool", "google.adk", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "research_assistant_app"
USER_ID = "local_user"
SESSION_ID = "default_session"

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")


# -----------------------------------------------------------------------------
# Helpers
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
    """
    Returns [(name, key)] via pyzotero; empty list if unavailable.
    """
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


def mcp_toolset(command: str, args: list[str], env: dict | None = None, timeout: float = 60.0):
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=command,
                args=args,
                env=env or {},
            ),
            timeout=timeout,
        )
    )


# Try both names for Zotero MCP (projects differ)
def zotero_mcp_toolset():
    env = {
        "ZOTERO_LIBRARY_ID": os.getenv("ZOTERO_USER_ID", ""),
        "ZOTERO_API_KEY": os.getenv("ZOTERO_API_KEY", ""),
        "ZOTERO_LIBRARY_TYPE": os.getenv("ZOTERO_LIBRARY_TYPE", "user"),
        "ZOTERO_LOCAL": os.getenv("ZOTERO_LOCAL", "false"),
    }
    return mcp_toolset("uvx", ["zotero-mcp"], env=env)


def ping_badge(ok: bool) -> str:
    return "✅" if ok else "❌"


# -----------------------------------------------------------------------------
# Streamlit caches
# -----------------------------------------------------------------------------
@st.cache_resource
def get_asyncio_runner() -> asyncio.Runner:
    return asyncio.Runner()


@st.cache_resource
def create_tracer():
    return OpikTracer(
        name="research-assistant-tracer",
        tags=["streamlit", "arxiv", "zotero", "mcp", "adk"],
        metadata={
            "environment": "development",
            "framework": "google-adk",
            "feature": "research-assistant",
        },
        project_name="adk-research-assistant",
    )


def create_retriever_agent(tracer, arxiv_ts, zotero_ts):
    instruction = """You are a retrieval agent that gathers many relevant papers.

Inputs arrive as CONFIG + user message. Follow this plan:

A) If CONFIG.use_zotero=true:
   1) Use Zotero MCP to enumerate seed items:
      - If CONFIG.zotero_collection_key present: list items in that collection (limit by CONFIG.zotero_recent).
      - Else if CONFIG.zotero_tag present: list items by tag (limit by CONFIG.zotero_recent).
   2) For each seed item, attempt (in order):
      a) Read the attached PDF (if accessible) to extract title/abstract (or first paragraph).
      b) If no attachment or unreadable, try to fetch the abstract from arXiv by title/DOI.
      c) Otherwise, fall back to Zotero metadata (title/authors/year) as a seed.
   3) Summarize seed themes in 1–2 sentences to guide expansion.

B) Expansion (ReAct-style; may call tools multiple times, up to CONFIG.search_iterations):
   - Formulate focused queries (AND/OR keywords, authors, categories) to arXiv.
   - Enforce CONFIG.cutoff if provided (exclude newer papers).
   - On each iteration: search, add new candidates, deduplicate (prefer DOI exact), and refine queries.
   - Stop when you reach ~CONFIG.top_k unique candidates or hit iteration limit.

Output strictly as JSON in a markdown code block with this schema:
{
  "papers": [
    {"source":"arxiv|zotero", "id":"...", "title":"...", "authors":["..."], "year": 2024,
     "doi":"", "url":"", "abstract":"(short)", "why_relevant":"(1-2 lines)"}
  ],
  "notes": "brief log of queries tried and iteration count"
}

Do not cluster. Keep abstracts short. Respect cutoffs. If Zotero fails, say so in 'notes' and proceed with arXiv-only."""

    return LlmAgent(
        model="gemini-2.5-flash-lite",
        name="retriever_agent",
        instruction=instruction,
        tools=[arxiv_ts, zotero_ts],  # both available
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
        before_tool_callback=tracer.before_tool_callback,
        after_tool_callback=tracer.after_tool_callback,
    )


def create_gap_agent(tracer):
    instruction = """You are a gap-analysis agent.

Given JSON with a list of retrieved papers (title, abstract, year, doi/url) and the original idea/context,
produce a concise gap analysis *without clustering*. Focus on:
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
Keep it tight and grounded to the provided papers. If list is small, say so and be conservative."""
    return LlmAgent(
        model="gemini-2.5-flash-lite",
        name="gap_agent",
        instruction=instruction,
        tools=[],  # no tools needed; just writes
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
    )


def create_review_agent(tracer):
    instruction = """You are a literature-review writing agent.

Given JSON with retrieved papers and the research idea/context, draft a literature review section
suitable for a paper. No clustering. Structure:

**Background & scope**: 1 paragraph tying the idea to major strands
**Synthesis of prior work**: 2–4 short paragraphs organized by themes/methods,
  weaving in citations by Title (Year) or DOI (avoid numeric citations).
**Methods & data seen**: 1 short paragraph (common study designs/datasets)
**Gaps motivating new work**: 1 paragraph (reference the earlier synthesis)

Keep it 400–700 words. Maintain academic tone, but crisp. Ground claims with references to specific papers from the provided list.
Cite papers by Title (Year) or DOI; avoid numeric citations. If the list is small, say so and be conservative.
"""
    return LlmAgent(
        model="gemini-2.5-flash-lite",
        name="review_agent",
        instruction=instruction,
        tools=[],
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
    )


def run_orchestrated_pipeline(user_prompt: str, config: dict) -> dict:
    """
    Returns dict with keys: retrieved_json (str), gap_md (str|None), review_md (str|None)
    """
    loop_runner = get_asyncio_runner()

    # 1) RETRIEVE
    retriever_content = types.Content(
        role="user",
        parts=[
            types.Part(
                text=("CONFIG:\n" + "\n".join(f"{k}={v}" for k, v in config.items()) + "\n\nUSER:\n" + user_prompt)
            )
        ],
    )

    retriever_text = loop_runner.run(
        _run_turn_async(
            st.session_state.runner_retriever,
            st.session_state.user_id,
            st.session_state.session_id + "_retriever",
            retriever_content,
        )
    )

    import re

    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", retriever_text)
    retrieved_json = m.group(1) if m else "{}"

    gap_md = None
    review_md = None

    # 2) GAP (optional)
    if config.get("generate_gap", False):
        gap_input = f"CONTEXT:\n{user_prompt}\n\nRETRIEVED_JSON:\n{retrieved_json}"
        gap_content = types.Content(role="user", parts=[types.Part(text=gap_input)])
        gap_md = loop_runner.run(
            _run_turn_async(
                st.session_state.runner_gap,
                st.session_state.user_id,
                st.session_state.session_id + "_gap",
                gap_content,
            )
        )

    # 3) REVIEW (optional)
    if config.get("generate_review", False):
        review_input = f"CONTEXT:\n{user_prompt}\n\nRETRIEVED_JSON:\n{retrieved_json}"
        review_content = types.Content(role="user", parts=[types.Part(text=review_input)])
        review_md = loop_runner.run(
            _run_turn_async(
                st.session_state.runner_review,
                st.session_state.user_id,
                st.session_state.session_id + "_review",
                review_content,
            )
        )

    return {"retrieved_json": retrieved_json, "gap_md": gap_md, "review_md": review_md}


@st.cache_resource
def create_agents_and_runners():
    tracer = create_tracer()

    # MCP toolsets
    arxiv_ts = mcp_toolset("uv", ["tool", "run", "arxiv-mcp-server"])
    zotero_ts = zotero_mcp_toolset()  # try both CLIs; ADK will connect to what works

    retriever = create_retriever_agent(tracer, arxiv_ts, zotero_ts)
    gap = create_gap_agent(tracer)
    review = create_review_agent(tracer)

    sess = InMemorySessionService()
    run_retriever = Runner(agent=retriever, app_name=APP_NAME, session_service=sess)
    run_gap = Runner(agent=gap, app_name=APP_NAME, session_service=sess)
    run_review = Runner(agent=review, app_name=APP_NAME, session_service=sess)

    asyncio.run(sess.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID + "_retriever"))
    asyncio.run(sess.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID + "_gap"))
    asyncio.run(sess.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID + "_review"))

    return (retriever, gap, review, run_retriever, run_gap, run_review)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📚 Research Assistant")
st.caption("Powered by Google ADK • arXiv MCP • Zotero MCP • Opik")

# Sidebar config
with st.sidebar:
    st.header("🔧 Mode & Sources")
    mode = st.radio("Start from", ["Idea", "Zotero"], index=0, horizontal=True)

    st.markdown("### Sources")
    use_zotero = st.checkbox("Use Zotero seeds", value=(mode == "Zotero"))

    # Collections UI
    selected_collection_key = ""
    selected_collection_name = ""

    if use_zotero:
        if zotero_env_ok():
            st.success(f"Zotero creds found {ping_badge(True)}")
            if HAVE_PYZOTERO:
                cols = list_zotero_collections()  # -> [(name, key), ...]
                if cols:
                    # selectbox returns the selected TUPLE; format_func just shows the name
                    selected = st.selectbox(
                        "Zotero collection",
                        options=cols,
                        format_func=lambda x: x[0],  # display name
                        index=0,
                    )
                    selected_collection_name, selected_collection_key = selected

                else:
                    st.warning("No collections found via pyzotero. You can still specify a tag below.")
                    selected_collection_name = ""
                    selected_collection_key = ""

            else:
                st.warning("pyzotero not installed; using text inputs instead.")
        else:
            st.error(f"Zotero not configured {ping_badge(False)} — set ZOTERO_USER_ID and ZOTERO_API_KEY")

        z_tag = st.text_input("Optional Zotero tag", placeholder="e.g. diffusion")
        z_recent = st.number_input("Recent N items", min_value=0, max_value=100, value=20, step=5)

    st.markdown("### Retrieval controls")
    top_k = st.slider("Top-k papers", min_value=10, max_value=100, value=30, step=5)
    cutoff = st.text_input("Date cutoff (YYYY-MM-DD, optional)", placeholder="e.g. 2023-12-31")
    clusters_k = st.slider("Clusters K", min_value=3, max_value=6, value=4, step=1)

    # in your sidebar under Retrieval controls
    search_iterations = st.slider("Search iterations (ReAct depth)", 1, 3, value=2, step=1)
    generate_gap = st.checkbox("Generate gap analysis", value=True)
    generate_review = st.checkbox("Generate literature review", value=False)

    st.markdown("---")
    st.markdown("### MCP health")
    # quick badges (best-effort; just prints env presence)
    st.caption(f"arXiv MCP: {ping_badge(True)} (assumes CLI available)")
    st.caption(f"Zotero MCP: {ping_badge(zotero_env_ok())}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.pop("messages", None)
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Session init (must run before using in run_orchestrated_pipeline) ----
if "user_id" not in st.session_state:
    st.session_state.user_id = USER_ID
if "session_id" not in st.session_state:
    st.session_state.session_id = SESSION_ID

# Multi-agent runners (create once)
if "runner_retriever" not in st.session_state:
    with st.spinner("Initializing multi-agent pipeline…"):
        (
            st.session_state.agent_retriever,
            st.session_state.agent_gap,
            st.session_state.agent_review,
            st.session_state.runner_retriever,
            st.session_state.runner_gap,
            st.session_state.runner_review,
        ) = create_agents_and_runners()


# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# Async run helper
async def _run_turn_async(runner, user_id, session_id, content):
    final_text_parts = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for p in event.content.parts:
                if getattr(p, "text", None):
                    final_text_parts.append(p.text)
        if event.is_final_response():
            break
    txt = "".join(final_text_parts).strip()
    return txt or "I couldn't produce a response this time."


# Chat input
prompt = st.chat_input("Type a research idea, topic, or a Zotero instruction…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    cfg = {
        "use_zotero": str(bool(use_zotero)).lower(),
        "zotero_collection_key": selected_collection_key or "",
        "zotero_collection_name": selected_collection_name or "",
        "zotero_tag": (z_tag if use_zotero else ""),
        "zotero_recent": (int(z_recent) if use_zotero else 0),
        "top_k": top_k,
        "cutoff": cutoff or "",
        "search_iterations": search_iterations,
        "generate_gap": generate_gap,
        "generate_review": generate_review,
    }

    with st.chat_message("assistant"):
        st.caption(
            f"Run config — Zotero: {use_zotero} • "
            f"Collection: {selected_collection_name or '—'} • "
            # f"Key: {selected_collection_key or '—'} • Tag: {z_tag or '—'} • "
            f"Iterations: {search_iterations} • Top-k: {top_k} • Cutoff: {cutoff or '—'}"
        )
        placeholder = st.empty()
        with st.spinner("Retrieving and synthesizing…"):
            out = run_orchestrated_pipeline(prompt, cfg)

        # Show outputs
        st.subheader("Retrieved set (JSON)")
        st.code(out["retrieved_json"], language="json")
        if generate_gap and out["gap_md"]:
            st.subheader("Gap analysis")
            st.markdown(out["gap_md"])
        if generate_review and out["review_md"]:
            st.subheader("Literature review (draft)")
            st.markdown(out["review_md"])

        # Save last assistant message summary into chat
        st.session_state.messages.append(
            {"role": "assistant", "content": "Completed retrieval and synthesis. See sections above."}
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Built with Streamlit • Google ADK • arXiv/Zotero MCP • Opik"
    "</div>",
    unsafe_allow_html=True,
)
