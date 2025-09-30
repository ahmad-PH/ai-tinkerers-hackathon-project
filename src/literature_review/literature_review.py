import asyncio
import logging
from typing import List, Dict, Any
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types
from mcp import StdioServerParameters
import os
import dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

class LiteratureReviewAgent:
    """
    A ReAct-based literature review agent that uses Google ADK and arXiv MCP server.
    
    The agent follows the ReAct pattern:
    1. Reason: Think about what information is needed
    2. Act: Use tools to gather information
    3. Observe: Process the results
    4. Repeat until sufficient information is gathered
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the literature review agent.
        
        Args:
            storage_path: Path to store downloaded papers (optional)
        """
        self.storage_path = storage_path or os.path.join(os.getcwd(), "papers")
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.agent = self._create_agent()
        self.runner = self._create_runner()
        
    def _create_agent(self) -> LlmAgent:
        """Create the ADK agent with ReAct-style instructions and arXiv MCP tools."""
        
        react_instruction = """
You are a literature review agent that follows the ReAct (Reasoning and Acting) pattern.

Your task is to help users find and analyze academic papers from arXiv. Follow this process:

1. **REASON**: Think about what the user is asking for and what information you need
2. **ACT**: Use the available tools to gather that information
3. **OBSERVE**: Process the results and determine if you have enough information
4. **REPEAT**: Continue reasoning and acting until you have sufficient information

Available tools:
- search_papers: Search for papers on arXiv
- download_paper: Download a specific paper
- list_papers: List downloaded papers
- read_paper: Read the content of a downloaded paper

When conducting a literature review:
1. Start by searching for relevant papers using search_papers
2. Analyze the search results to identify the most relevant papers
3. Download and read key papers if needed for deeper analysis
4. Synthesize the findings into a comprehensive literature review

Always format your responses clearly with:
- Paper titles and authors
- Publication dates
- Key findings and contributions
- Relevance to the user's query
- Links to arXiv papers

Be thorough but concise. Focus on the most relevant and recent papers.
"""
        
        return LlmAgent(
            model="gemini-2.0-flash",
            name="literature_review_agent",
            instruction=react_instruction,
            tools=[
                McpToolset(
                    connection_params=StdioConnectionParams(
                        server_params=StdioServerParameters(
                            command="uv",
                            args=[
                                "tool", 
                                "run", 
                                "arxiv-mcp-server",
                                "--storage-path",
                                os.path.abspath(self.storage_path)
                            ],
                        ),
                        timeout=60.0,
                    ),
                    # Filter to only the core arXiv tools
                    tool_filter=["search_papers", "download_paper", "list_papers", "read_paper"],
                )
            ],
        )
    
    def _create_runner(self) -> Runner:
        """Create the ADK runner with session management."""
        session_service = InMemorySessionService()
        runner = Runner(
            agent=self.agent,
            app_name="literature_review_app",
            session_service=session_service,
        )
        
        # Create session
        asyncio.run(
            session_service.create_session(
                app_name="literature_review_app",
                user_id="user",
                session_id="default"
            )
        )
        
        return runner
    
    async def _run_query_async(self, query: str) -> str:
        """Run a query asynchronously and return the response."""
        try:
            final_text_parts = []
            content = types.Content(role="user", parts=[types.Part(text=query)])
            
            async for event in self.runner.run_async(
                user_id="user",
                session_id="default", 
                new_message=content,
            ):
                # Collect text parts
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "text", None):
                            final_text_parts.append(part.text)
                
                # Stop when we get the final response
                if event.is_final_response():
                    break
            
            return "".join(final_text_parts).strip()
            
        except Exception as e:
            logger.error(f"Error running query: {e}")
            return f"Error processing query: {str(e)}"
    
    def literature_review(self, query: str) -> str:
        """
        Conduct a literature review for the given query.
        
        Args:
            query: The research question or topic to investigate
            
        Returns:
            A formatted literature review with relevant papers and analysis
        """
        logger.info(f"Starting literature review for query: {query}")
        
        # Run the async query
        runner = asyncio.Runner()
        result = runner.run(self._run_query_async(query))
        
        logger.info("Literature review completed")
        return result


def literature_review(query: str) -> str:
    """
    Simple function interface for conducting literature reviews.
    
    Args:
        query: The research question or topic to investigate
        
    Returns:
        A formatted literature review with relevant papers and analysis
    """
    agent = LiteratureReviewAgent()
    return agent.literature_review(query)