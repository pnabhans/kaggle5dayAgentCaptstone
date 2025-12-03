import os
import vertexai
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.cloud import secretmanager
import google.auth

# Initialize Vertex AI
vertexai.init(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
)

def get_secret(secret_name, project_id):
    """Fetches secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# --- Agent Configuration ---
# We do NOT fetch the secret at the global level.
# We fetch it inside the tool setup to keep import safe.

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

personal_assistant = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="concierge",
    instruction="""You are a routing agent
    Given a request to travel plan
    Identify the start and end locations, find a travel route where mode of travel can be by vehicle on road
    or by air. If a locally preferred and reliable public transport is available include that
    """,
    tools=[google_search] 
)

# This function is the bridge. It runs LOCALLY to prepare the agent,
# then the fully prepared agent (with key inside) is sent to the cloud.
async def on_startup():
    print("🔐 Fetching Maps API Key from Secret Manager...")
    
    # 1. Fetch Secret
    # Note: This runs on your machine/notebook first. 
    # Ensure your local user has 'Secret Manager Secret Accessor' role too!
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") 
    maps_key = get_secret("mcp-map-api-key", project_id)
    
    # 2. Configure Tools
    print("⚙️  Initializing MCP Tools...")
    maps_mcp = McpToolset(
        connection_params=StreamableHTTPServerParams(
            url="https://mapstools.googleapis.com/mcp",
            headers={"X-Goog-Api-Key": maps_key}
        )
    )
    
    # 3. Attach to Agent
    tool_list = await maps_mcp.get_tools()
    desired_tools = ["search_places", "compute_routes"]
    filtered_tools = [t for t in tool_list if t.name in desired_tools]
    
    personal_assistant.tools.extend(filtered_tools)
    print(f"✅ Agent ready with {len(personal_assistant.tools)} tools.")
    
    return personal_assistant
