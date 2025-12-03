import sys
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from dotenv import load_dotenv # Add this
load_dotenv()

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
active_mcp_toolsets = []
async def get_mcp_map_tools():
    maps_api_key = os.getenv("MAP_API_KEY")
    print(f"Maps API Key: {maps_api_key}")
    desired_tools = ["search_places", "compute_routes"]
    maps_mcp = McpToolset(
    connection_params=StreamableHTTPServerParams(
        url="https://mapstools.googleapis.com/mcp",
        headers={"X-Goog-Api-Key": maps_api_key} if maps_api_key else {}
    )
    )
    active_mcp_toolsets.append(maps_mcp) 
    
    tool_list = await maps_mcp.get_tools() 
    print(f"Successfully loaded {len(tool_list)} tools.")

    # 4. Filter the resulting list synchronously
    filtered_tools_list = [
        tool for tool in tool_list
        if tool.name in desired_tools
    ]
    return filtered_tools_list

root_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="personal_assistant",
    instruction="""You are a travel assistant focused on getting direction to a place.
                    You have the following tools at your disposal:
                    1. MCP Tools (Search Places, Compute Routes and Weather lookup)""",
    description="A concierge agent that helps with daily tasks.",
    tools=[google_search]
)

async def initialize_agent(agent):
    """Initializes the agent by asynchronously loading and setting the MCP tools."""
    
    # AWAIT the tool loading function
    map_tools = await get_mcp_map_tools()
    
    # 4. Update the agent's tools list with the loaded MCP tools
    # agent.tools = agent.tools +map_tools # 
    agent.tools = map_tools # Append the new tools to the existing ones
    print(f"Agent '{agent.name}' successfully initialized with {len(agent.tools)} tools.")
    
    # Return the initialized agent instance
    return agent
