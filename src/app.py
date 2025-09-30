# app.py
import asyncio
import re
import hashlib
import json
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

load_dotenv()  # Load environment variables from .env file if present

APP_NAME = "research_assistant_app"
USER_ID = "local_user"
SESSION_ID = "default_session"

# Page config
st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")


# Initialize agent (cached to avoid recreating on every rerun)
@st.cache_resource
def create_agent():
    """Create the ADK agent with arXiv MCP toolset"""
    agent = LlmAgent(
        model='gemini-2.5-flash-lite',
        name='research_assistant',
        instruction="""You are a research assistant that helps users find academic papers.

When the user provides a research topic or question:
1. Use the arXiv search tools to find relevant papers
2. Present the results in a clear, organized format
3. Include: title, authors, publication date, abstract summary, and arXiv link
4. Prioritize recent papers (last 2-3 years) when relevant
5. If the user asks for a specific number of papers, try to provide that many

Format your responses as markdown for better readability.""",
        tools=[
            McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command='uv',
                        args=['tool', 'run', 'arxiv-mcp-server'],
                    ),
                    timeout=60.0,
                )
            )
        ],
    )
    return agent


@st.cache_resource
def create_runner():
    # use cached agent, so this does not re-create
    agent = create_agent()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # 👇 CREATE the session once (sync-friendly)
    asyncio.run(session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID))

    return runner


# --- Helpers to render papers nicely in separate boxes ---
def _split_markdown_into_papers(markdown_text: str) -> list[str]:
    """Best-effort segmentation of assistant markdown into paper-sized chunks.

    Heuristics:
    - Split on blank lines that precede common paper starters: headings (###/##),
      numbered items (1.), bullets (- or *), or explicit "Title:" markers.
    - If nothing reasonable is found, return a single chunk.
    """
    if not markdown_text:
        return []

    text = markdown_text.strip()

    # Try splitting where a new paper likely starts
    chunks = re.split(r"\n\s*\n(?=\s*(?:#{2,3}\s|\d+\.\s|[-*]\s|Title:))", text)

    # Post-filter: keep meaningful chunks
    cleaned = [c.strip() for c in chunks if c and c.strip()]

    # If the split was too aggressive or not helpful, fallback to a simpler rule:
    if len(cleaned) <= 1:
        alt = re.split(r"\n\s*\n\s*\n+", text)  # split on big paragraph gaps
        alt_cleaned = [c.strip() for c in alt if c and c.strip()]
        if len(alt_cleaned) > 1:
            cleaned = alt_cleaned

    # Final sanity: if still just one, return as single block
    return cleaned if cleaned else [text]


def extract_title_from_markdown(paper_md: str) -> str:
    """Extract a likely title from a paper markdown block.

    Priority:
    - First heading line starting with ### or ##
    - Line starting with Title:
    - First non-empty line (trimmed)
    """
    lines = [l.strip() for l in paper_md.splitlines() if l.strip()]
    for l in lines:
        if l.startswith("### ") or l.startswith("## "):
            return normalize_title(l.lstrip('#').strip())
    for l in lines:
        if l.lower().startswith("title:"):
            return normalize_title(l.split(":", 1)[1].strip() or "Untitled")
    return normalize_title(lines[0]) if lines else "Untitled"


def normalize_title(raw: str) -> str:
    """Remove markdown syntax and ordering from a title-like string.

    - Strip link markup: [text](url) -> text
    - Remove leading heading hashes, list markers, and numeric prefixes (e.g., "1. ")
    - Remove common emphasis/backtick wrappers and collapse whitespace
    """
    if not raw:
        return "Untitled"

    s = raw.strip()
    # Remove link markdown
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\\1", s)
    # Remove heading markers (if any remain)
    s = s.lstrip('#').strip()
    # Remove list/numeric prefixes like "1. ", "- ", "* ", "+ "
    s = re.sub(r"^(?:\d+\.\s+|[-*+]\s+)", "", s)
    # Strip surrounding emphasis/backticks
    s = s.strip("*_` ")
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s or "Untitled"


def _paper_key_from_markdown(paper_md: str) -> str:
    """Generate a stable key for a paper based on content hash."""
    digest = hashlib.sha256(paper_md.strip().encode("utf-8")).hexdigest()
    return f"paper_{digest[:16]}"


def _render_assistant_response(markdown_text: str, namespace: str) -> None:
    """Render assistant markdown as paper boxes with like/dislike.

    The namespace makes Streamlit widget keys stable and unique across messages.
    """
    papers = _split_markdown_into_papers(markdown_text)
    if len(papers) <= 1:
        st.markdown(markdown_text)
        return

    for idx, paper_md in enumerate(papers, start=1):
        with st.container(border=True):
            st.markdown(paper_md)

            key_core = _paper_key_from_markdown(paper_md)
            key = f"{namespace}_{key_core}"
            if key not in st.session_state.paper_feedback:
                # Extract a human-friendly title for export purposes
                title = extract_title_from_markdown(paper_md)
                # Initialize entry
                st.session_state.paper_feedback[key] = {
                    'likes': 0,
                    'dislikes': 0,
                    'vote': None,
                    'title': title,
                }
                # If we have a loaded score for this title, apply it once
                if 'title_scores' in st.session_state and title in st.session_state.title_scores:
                    score = st.session_state.title_scores[title]
                    if score == 1:
                        st.session_state.paper_feedback[key]['likes'] = 1
                        st.session_state.paper_feedback[key]['dislikes'] = 0
                        st.session_state.paper_feedback[key]['vote'] = 'like'
                    elif score == -1:
                        st.session_state.paper_feedback[key]['likes'] = 0
                        st.session_state.paper_feedback[key]['dislikes'] = 1
                        st.session_state.paper_feedback[key]['vote'] = 'dislike'
                    else:
                        st.session_state.paper_feedback[key]['likes'] = 0
                        st.session_state.paper_feedback[key]['dislikes'] = 0
                        st.session_state.paper_feedback[key]['vote'] = None

            cols = st.columns([1, 1, 6])
            with cols[0]:
                current_vote = st.session_state.paper_feedback[key].get('vote')
                like_disabled = current_vote == 'like'
                if st.button("👍 Like", key=f"like_{key}", disabled=like_disabled):
                    # Only count if switching from no vote or from dislike
                    if current_vote == 'dislike':
                        st.session_state.paper_feedback[key]['dislikes'] = max(
                            0, st.session_state.paper_feedback[key]['dislikes'] - 1
                        )
                    if current_vote != 'like':
                        st.session_state.paper_feedback[key]['likes'] += 1
                        st.session_state.paper_feedback[key]['vote'] = 'like'
            with cols[1]:
                current_vote = st.session_state.paper_feedback[key].get('vote')
                dislike_disabled = current_vote == 'dislike'
                if st.button("👎 Dislike", key=f"dislike_{key}", disabled=dislike_disabled):
                    # Only count if switching from no vote or from like
                    if current_vote == 'like':
                        st.session_state.paper_feedback[key]['likes'] = max(
                            0, st.session_state.paper_feedback[key]['likes'] - 1
                        )
                    if current_vote != 'dislike':
                        st.session_state.paper_feedback[key]['dislikes'] += 1
                        st.session_state.paper_feedback[key]['vote'] = 'dislike'
            with cols[2]:
                fb = st.session_state.paper_feedback[key]
                st.caption(f"Likes: {fb['likes']} • Dislikes: {fb['dislikes']}")


def save_feedback_to_file(path: str) -> tuple[bool, str]:
    """Save feedback as { title: { score } } with score in {1,0,-1}."""
    try:
        export: dict[str, dict] = {}
        for entry in st.session_state.paper_feedback.values():
            title = normalize_title(entry.get('title') or 'Untitled')
            vote = entry.get('vote')
            if vote == 'like':
                score = 1
            elif vote == 'dislike':
                score = -1
            else:
                score = 0
            export[title] = {"score": score}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        return True, f"Saved feedback to {path}"
    except Exception as e:
        return False, str(e)


def load_feedback_from_file(path: str) -> tuple[bool, str]:
    """Load { title: { score } }, store in session as title->score for later binding."""
    try:
        if not os.path.exists(path):
            return False, "File does not exist"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return False, "Invalid file format"

            # Convert into a simpler mapping: title -> score
            title_scores: dict[str, int] = {}
            for title, obj in data.items():
                title = normalize_title(title)
                if isinstance(obj, dict) and "score" in obj:
                    score = obj["score"]
                    if score in (-1, 0, 1):
                        title_scores[title] = score
            st.session_state.title_scores = title_scores
        return True, f"Loaded feedback from {path}"
    except Exception as e:
        return False, str(e)


# UI Header
st.title("📚 Research Assistant")
st.markdown("*Powered by ADK + arXiv MCP*")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'paper_feedback' not in st.session_state:
    st.session_state.paper_feedback = {}

if 'feedback_file' not in st.session_state:
    # default to a file next to this app
    st.session_state.feedback_file = os.path.join(os.path.dirname(__file__), "paper_feedback.json")

if 'runner' not in st.session_state:
    with st.spinner("Initializing research assistant..."):
        st.session_state.agent = create_agent()
        st.session_state.runner = create_runner()
        st.session_state.user_id = USER_ID
        st.session_state.session_id = SESSION_ID

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_assistant_response(message["content"], namespace=f"hist_{idx}")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What research topic would you like to explore?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Searching arXiv..."):
            try:
                # Prepare ADK content
                content = types.Content(role="user", parts=[types.Part(text=prompt)])
                final_text = None
                for event in st.session_state.runner.run(
                    user_id=st.session_state.user_id, session_id=st.session_state.session_id, new_message=content
                ):
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            final_text = event.content.parts[0].text
                        break

                if not final_text:
                    final_text = "I couldn't produce a response this time."

                # Display response consistently using the same renderer as history
                with message_placeholder.container():
                    _render_assistant_response(final_text, namespace="live")

                # Add assistant response to chat history (store original markdown)
                st.session_state.messages.append({"role": "assistant", "content": final_text})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your query or check that the arXiv MCP server is working correctly."
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Sidebar with example queries and info
with st.sidebar:
    st.header("📖 Quick Start")

    st.markdown("### Example Queries")

    example_queries = [
        "Find recent papers on transformer architectures",
        "Show me 5 papers about federated learning privacy",
        "What are the latest developments in vision transformers?",
        "Search for papers on reinforcement learning from human feedback",
    ]

    for query in example_queries:
        if st.button(query, key=query):
            # Trigger a rerun with the example query
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    st.markdown("---")

    st.markdown("### How to Use")
    st.markdown("""
    1. Enter your research topic or question
    2. The assistant will search arXiv
    3. View organized results with paper details
    4. Click on arXiv links to read full papers
    """)

    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This assistant uses:
    - **Google ADK** for agent orchestration
    - **arXiv MCP Server** for paper search
    - **Gemini 2.0 Flash** for intelligent responses
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Feedback Counts"):
        st.session_state.paper_feedback = {}
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Save/Load Feedback")
    st.session_state.feedback_file = st.text_input("Feedback file path", st.session_state.feedback_file)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💾 Save Feedback"):
            ok, msg = save_feedback_to_file(st.session_state.feedback_file)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    with col_b:
        if st.button("📂 Load Feedback"):
            ok, msg = load_feedback_from_file(st.session_state.feedback_file)
            if ok:
                st.success(msg)
                st.experimental_rerun()
            else:
                st.error(msg)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Built with Streamlit • Powered by Google ADK"
    "</div>",
    unsafe_allow_html=True,
)
