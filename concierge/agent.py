import os
import vertexai
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
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
class TravelConciergeApp(LlmAgent):

    mcp_client: McpToolset | None = None
    def __init__(self):
        """Initializes the Agent synchronously, passing all configuration to the base class."""
        
        # 1. Define Model (Local variable, used for super().__init__ only)
        model = Gemini(model="gemini-2.0-flash")

        # 2. CRUCIAL FIX: Call the parent constructor (LlmAgent) FIRST!
        # This initializes the Pydantic state and makes *THIS* object the Root Agent.
        super().__init__(
            model=model,
            name="RootAgent",
            instruction="You are a travel assistant and must use the loaded MCP tools.",
            tools=[] # Tools are initially empty, to be loaded in set_up.
        )
        
        # 3. ELIMINATE REDUNDANCY: Remove 'self.agent = LlmAgent(...)'. 
        # The 'self' object *is* the agent now.
       


    async def set_up(self):
        """
        THE HOOK: Vertex AI calls this async method once at deployment startup.
        We inject tools into the inherited list (self.tools).
        """
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
                
                # 2. Fetch the Tools (Async)
                all_tools = await self.mcp_client.get_tools()
                
                # 3. Inject into the inherited tools list (self.tools)
                self.tools.extend(all_tools)
                print(f"✅ HOOK COMPLETE: Injected {len(all_tools)} MCP tools.")
                
            except Exception as e:
                print(f"❌ HOOK FAILED: Could not connect to MCP server: {e}")

    async def query(self, message: str):
        """Handles synchronous/single response queries by running THIS object."""
        # Fix: Call the inherited run_async method on self
        response = await self.run_async(message)
        return response.text

    async def stream_query(self, message: str):
        """Handles asynchronous streaming queries by running THIS object."""
        # Fix: Call the inherited run_stream method on self
        response_stream = self.run_stream(message)
        async for chunk in response_stream:
            yield chunk.text

# --- FINAL EXPORT ---
root_agent = TravelConciergeApp()
