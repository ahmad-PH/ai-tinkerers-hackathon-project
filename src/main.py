# ./adk_agent_samples/mcp_agent/agent.py
import os

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters

# Where PDFs/markdown should be cached by arxiv-mcp-server.
# Use an ABSOLUTE path; here we make a "papers" folder next to this file.
STORAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
os.makedirs(STORAGE_FOLDER, exist_ok=True)  # ensure it exists

# NOTE: arxiv-mcp-server exposes tools like:
#   - search_papers, download_paper, list_papers, read_paper
# (we optionally filter to just those four)
# Ref: README tool list and config show running via `uv tool run arxiv-mcp-server`
# with `--storage-path ...` argument.
root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="arxiv_research_assistant",
    instruction=(
        "You help the user find and digest arXiv papers. Use the MCP tools to search, fetch, list, and read papers."
    ),
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    # arxiv-mcp-server is packaged for uv; this starts it over stdio:
                    #   uv tool run arxiv-mcp-server --storage-path <ABS_PATH>
                    # You can also set ARXIV_STORAGE_PATH env var instead of --storage-path.
                    command="uvx",
                    args=[
                        "tool",
                        "run",
                        "arxiv-mcp-server",
                        "--storage-path",
                        os.path.abspath(STORAGE_FOLDER),
                    ],
                ),
            ),
            # Limit to the core tools exposed by this server (optional)
            tool_filter=["search_papers", "download_paper", "list_papers", "read_paper"],
        )
    ],
)


def main():
    # Depending on your ADK runtime you may hand this agent to a runner.
    # If you're embedding it into a larger app, import `root_agent` from this module.
    print("arxiv_research_assistant is configured. Hand off to your ADK runtime/loop.")


if __name__ == "__main__":
    main()
