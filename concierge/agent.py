from google.adk.agents import Agent
import vertexai
import os
from google.adk.models import Gemini
from google.genai import types
# MCP Imports
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.google_search_tool import google_search



vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

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
model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config) 
# maps_api_key = os.getenv("MAP_API_KEY")
  
# if not maps_api_key:
#     print("WARNING: MAP_API_KEY not found in environment variables.")
#     return
# maps_mcp = McpToolset(
#     connection_params=StreamableHTTPServerParams(
#         url="https://mapstools.googleapis.com/mcp",
#         headers={"X-Goog-Api-Key": maps_api_key}
#     )
# )

root_agent = Agent(
    model=model,
    name="personal_assistant",
    instruction="""You are a travel assistant.
                   Use the provided MCP tools to Search Places and Compute Routes.
                   Always verify the location before calculating a route.""",
    tools=[google_search]
)

