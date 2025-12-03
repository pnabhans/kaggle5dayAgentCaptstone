import os
import sys
import asyncio
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types

# MCP Imports
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams

# 1. Setup Configuration
from dotenv import load_dotenv 
load_dotenv()

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# 2. Define the Agent
model = Gemini(model="gemini-1.5-flash-002", retry_options=retry_config) 

root_agent = Agent(
    model=model,
    name="personal_assistant",
    instruction="""You are a travel assistant.
                   Use the provided MCP tools to Search Places and Compute Routes.
                   Always verify the location before calculating a route.""",
    tools=[google_search] 
)

# 3. Define the Startup Logic
active_mcp_toolsets = []

async def startup_sequence(app_state: Agent):
    """
    This function runs ONCE when the first query is received.
    It connects to MCP, gets tools, and attaches them to the agent.
    """
    print("--- Starting MCP Initialization ---")
    maps_api_key = os.getenv("MAP_API_KEY")
    
    if not maps_api_key:
        print("WARNING: MAP_API_KEY not found in environment variables.")
        return

    maps_mcp = McpToolset(
        connection_params=StreamableHTTPServerParams(
            url="https://mapstools.googleapis.com/mcp",
            headers={"X-Goog-Api-Key": maps_api_key}
        )
    )
    
    # Store reference so it doesn't get garbage collected
    active_mcp_toolsets.append(maps_mcp)
    
    try:
        tool_list = await maps_mcp.get_tools()
        print(f"MCP Loaded: {len(tool_list)} tools.")
        
        desired_tools = ["search_places", "compute_routes"]
        filtered_tools = [t for t in tool_list if t.name in desired_tools]
        
        # Append the new tools to the existing list safely
        app_state.tools.extend(filtered_tools)
        
        print(f"Agent initialized. Total tools: {len(app_state.tools)}")
    except Exception as e:
        print(f"Failed to initialize MCP: {e}")


class AgentEngine:
    def __init__(self, agent: Agent):
        self.agent = agent
        self._startup_fn = None
        self._is_initialized = False

    def register_startup(self, fn):
        """Registers the async function to run before the first query."""
        self._startup_fn = fn

    async def query(self, message: str, **kwargs):
        """
        The method Vertex AI calls. 
        """
        # 1. Lazy Initialization (Run startup logic if it hasn't run yet)
        if self._startup_fn and not self._is_initialized:
            print("First request detected: Running startup sequence...")
            await self._startup_fn(self.agent)
            self._is_initialized = True

        # 2. Run the actual Agent
        # Note: Depending on your exact ADK version, this is usually .run() or .run_async()
        # The ADK Agent handles the conversation loop.
        response = await self.agent.run(message)
        
        # 3. Return string result (Vertex expects a string or valid JSON)
        return response.text

# 5. Create the Engine App
# Vertex AI will load this object
app = AgentEngine(root_agent)

# 6. Register the Hook
app.register_startup(startup_sequence)
