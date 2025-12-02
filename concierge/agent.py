import os
import sys

import vertexai
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams

from google.cloud import secretmanager
from google.api_core.exceptions import GoogleAPICallError


vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)
def get_secret(secret_env_name: str, project_id: str = None, secret_manager_name: str = None):
    """
    Tries to get a secret from environment variables first (local dev).
    If not found, falls back to Google Cloud Secret Manager (production).
    """
    # 1. Try Local Environment Variable (.env)
    secret_value = os.environ.get(secret_env_name)
    if secret_value:
        print(f"Loaded {secret_env_name} from local environment.")
        return secret_value

    # 2. Fallback to Secret Manager (Production)
    print(f"{secret_env_name} not found locally. Attempting Secret Manager...")
    
    if not project_id or not secret_manager_name:
         raise ValueError(f"Cannot look up secret: GCP Project ID or Secret Name missing.")

    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_manager_name}/versions/latest"

    try:
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        print(f"Successfully loaded secret from Secret Manager: {secret_manager_name}")
        return secret_value
    except GoogleAPICallError as e:
        print(f"Error fetching secret from Secret Manager: {e}")
        # Re-raise so the agent fails to start if critical key is missing
        raise


PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
MAPS_SECRET_NAME_IN_GCP = "mcp-map-api-key" 
# Get mcp api key
maps_api_key=None
try:
    maps_api_key = get_secret(
        secret_env_name="GOOGLE_API_KEY", 
        project_id=PROJECT_ID, 
        secret_manager_name=MAPS_SECRET_NAME_IN_GCP
    )
except Exception as e:
     print("CRITICAL: Failed to acquire Maps API Key. Agent cannot start.")
     raise


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
active_mcp_toolsets = []
async def get_mcp_map_tools():
    maps_api_key = os.getenv("GOOGLE_API_KEY")
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

personal_assistant = Agent(
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
