# app.py
import asyncio
import logging

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

configure()

load_dotenv()  # Load environment variables from .env file if present

# Reduce noise from ADK/MCP internals; keep warnings+errors
for noisy in ("mcp", "google.adk.tools.mcp_tool", "google.adk", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "research_assistant_app"
USER_ID = "local_user"
SESSION_ID = "default_session"

# Page config
st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def get_asyncio_runner() -> asyncio.Runner:
    # One runner (and event loop) for the lifetime of the Streamlit session
    return asyncio.Runner()


@st.cache_resource
def create_tracer():
    # You can tweak these fields (tags, metadata, project_name) to your needs.
    return OpikTracer(
        name="research-assistant-tracer",
        tags=["streamlit", "arxiv", "mcp", "adk"],
        metadata={
            "environment": "development",
            "framework": "google-adk",
            "feature": "research-assistant",
        },
        project_name="adk-research-assistant",
    )


# Initialize agent (cached to avoid recreating on every rerun)
@st.cache_resource
def create_agent():
    """Create the ADK agent with arXiv MCP toolset"""
    tracer = create_tracer()

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


# UI Header
st.title("📚 Research Assistant")
st.markdown("*Powered by ADK + arXiv MCP*")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'runner' not in st.session_state:
    with st.spinner("Initializing research assistant..."):
        st.session_state.agent = create_agent()
        st.session_state.runner = create_runner()
        st.session_state.user_id = USER_ID
        st.session_state.session_id = SESSION_ID

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


async def _run_turn_async(runner, user_id, session_id, content):
    final_text_parts = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        # collect all text so we don't warn about non-text parts
        if event.content and event.content.parts:
            for p in event.content.parts:
                if getattr(p, "text", None):
                    final_text_parts.append(p.text)
        if event.is_final_response():
            break
    txt = "".join(final_text_parts).strip()
    return txt or "I couldn't produce a response this time."


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
                content = types.Content(role="user", parts=[types.Part(text=prompt)])
                runner = get_asyncio_runner()
                final_text = runner.run(
                    _run_turn_async(
                        st.session_state.runner,
                        st.session_state.user_id,
                        st.session_state.session_id,
                        content,
                    )
                )

                message_placeholder.markdown(final_text)
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

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Built with Streamlit • Powered by Google ADK"
    "</div>",
    unsafe_allow_html=True,
)
