import os
import sys
import asyncio
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.adk.engine import AgentEngine # Import AgentEngine
from google.genai import types

# MCP Imports
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams

# 1. Setup Configuration
# Note: In Vertex AI, ENV variables are set via the console or deployment parameters, 
# but load_dotenv is fine for local testing.
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
# Note: "gemini-2.5-flash-lite" does not exist publicly yet. 
# Use gemini-1.5-flash or gemini-1.5-pro for now.

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
    This function runs ONCE when the container starts.
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
        
       
        app_state.tools=filtered_tools
        
        print(f"Agent initialized. Total tools: {len(app_state.tools)}")
    except Exception as e:
        print(f"Failed to initialize MCP: {e}")

# 4. Create the Engine App
app = AgentEngine(root_agent)

# 5. Register the Hook
app.register_startup(startup_sequence)
