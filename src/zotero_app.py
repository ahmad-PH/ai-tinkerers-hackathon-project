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


@st.cache_resource
def create_agent():
    tracer = create_tracer()

    # MCP toolsets
    arxiv_ts = mcp_toolset("uv", ["tool", "run", "arxiv-mcp-server"])
    zotero_ts = zotero_mcp_toolset()  # try both CLIs; ADK will connect to what works

    instruction = """You are a research assistant that helps users find and synthesize academic papers.

Base workflow:
1) If CONFIG.use_zotero is true:
   a) Use Zotero MCP to list/retrieve seed items from the specified collection key or by tag (limit by CONFIG.zotero_recent).
   b) Extract titles/DOIs/abstracts; summarize seed themes in 1-2 sentences.
   c) Expand with arXiv search using focused queries derived from seeds.
2) If CONFIG.use_zotero is false, search arXiv directly.
3) Present standardized outputs:
   - Shortlist (title, authors, year, link) for top-k
   - 3–5 clusters with 1-liner each and anchor papers
   - 4–8 gap bullets with evidence refs (paper IDs)
4) Always format in markdown. If a step fails (e.g., Zotero), fall back and say so.
"""

    agent = LlmAgent(
        model="gemini-2.5-flash-lite",
        name="research_assistant",
        instruction=instruction,
        tools=[arxiv_ts, zotero_ts],
        before_agent_callback=tracer.before_agent_callback,
        after_agent_callback=tracer.after_agent_callback,
        before_model_callback=tracer.before_model_callback,
        after_model_callback=tracer.after_model_callback,
        before_tool_callback=tracer.before_tool_callback,
        after_tool_callback=tracer.after_tool_callback,
    )
    return agent


@st.cache_resource
def create_runner():
    agent = create_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    asyncio.run(session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID))
    return runner


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

if "runner" not in st.session_state:
    with st.spinner("Initializing agent…"):
        st.session_state.agent = create_agent()
        st.session_state.runner = create_runner()
        st.session_state.user_id = USER_ID
        st.session_state.session_id = SESSION_ID

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
    # Add user msg
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build CONFIG + SYSTEM_HINT for deterministic behavior
    cfg_lines = [
        f"use_zotero={str(bool(use_zotero)).lower()}",
        f"zotero_collection_key={selected_collection_key or ''}",
        f"zotero_collection_name={selected_collection_name or ''}",
        f"zotero_tag={z_tag if use_zotero else ''}",
        f"zotero_recent={(int(z_recent) if use_zotero else 0)}",
        f"top_k={top_k}",
        f"cutoff={cutoff or ''}",
        f"clusters_k={clusters_k}",
    ]
    config_blob = "CONFIG:\n" + "\n".join(cfg_lines) + "\n"

    system_hint = """SYSTEM_HINT:
When CONFIG.use_zotero=true:
1) Use Zotero MCP to fetch seeds: if CONFIG.zotero_collection_key, list items from that collection; else if CONFIG.zotero_tag, list items by tag; limit by CONFIG.zotero_recent.
2) Extract titles/abstracts/DOIs from seeds; summarize seed themes succinctly.
3) Expand with arXiv queries derived from seeds; enforce CONFIG.cutoff if present.
4) Return: shortlist (size=CONFIG.top_k), 3–5 clusters (use CONFIG.clusters_k as a hint), gap bullets (each cites paper IDs).
If Zotero fails or returns nothing, fall back to arXiv-only and state the fallback."""

    user_text = f"{system_hint}\n{config_blob}\n{prompt}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            try:
                content = types.Content(role="user", parts=[types.Part(text=user_text)])
                loop_runner = get_asyncio_runner()
                final_text = loop_runner.run(
                    _run_turn_async(
                        st.session_state.runner,
                        st.session_state.user_id,
                        st.session_state.session_id,
                        content,
                    )
                )
                placeholder.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                err = f"❌ Error: {e}\n\nIf Zotero is on, verify MCP server and env vars. Otherwise try again."
                placeholder.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Built with Streamlit • Google ADK • arXiv/Zotero MCP • Opik"
    "</div>",
    unsafe_allow_html=True,
)
