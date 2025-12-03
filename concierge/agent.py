import os
import vertexai
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ToolContext # ToolContext is optional but good practice
from google.adk.models import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams

# --- CONFIGURATION & INFRASTRUCTURE ---
load_dotenv()
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

# --- THE AGENT HOST CLASS (The Framework Glue) ---
class TravelConciergeApp:
    def __init__(self):
        """Initializes the Agent synchronously, with an empty tool list."""
        self.model = Gemini(model="gemini-1.5-flash")
        
        self.agent = LlmAgent(
            model=self.model,
            name="travel_assistant",
            instruction="You are a travel assistant. Use the provided MCP tools to search places and compute routes.",
            tools=[] # Start empty! Tools loaded in set_up.
        )
        # Store MCP reference to prevent garbage collection
        self.mcp_client = None

    async def set_up(self):
        """
        THE HOOK: Vertex AI calls this async method once at deployment startup.
        This is where we safely connect to the MCP server.
        """
        print("🪝 HOOK: Starting Async MCP Setup...")
        maps_key = os.getenv("MAP_API_KEY")
        
        if maps_key:
            try:
                # 1. Instantiate the Toolset Client
                self.mcp_client = McpToolset(
                    connection_params=StreamableHTTPServerParams(
                        url="https://mapstools.googleapis.com/mcp",
                        headers={"X-Goog-Api-Key": maps_key}
                    )
                )
                
                # 2. Fetch the Tools (This requires 'await')
                # No filtering: we get everything MCP provides.
                all_tools = await self.mcp_client.get_tools()
                
                # 3. Inject into the Agent
                self.agent.tools.extend(all_tools)
                print(f"✅ HOOK COMPLETE: Injected {len(all_tools)} MCP tools.")
                
            except Exception as e:
                print(f"❌ HOOK FAILED: Could not connect to MCP server: {e}")
        else:
            print("⚠️ WARNING: MAP_API_KEY missing. Agent running without maps.")

    async def query(self, message: str):
        """Handles synchronous/single response queries."""
        response = await self.agent.run_async(message)
        return response.text

    async def stream_query(self, message: str):
        """Handles asynchronous streaming queries (for Kaggle testing)."""
        response_stream = self.agent.run_stream(message)
        async for chunk in response_stream:
            yield chunk.text

# No global root_agent = ... or query function needed outside the class!
