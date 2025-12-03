import os
import asyncio
import vertexai
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.google_search_tool import google_search

# --- INFRASTRUCTURE SETUP ---
load_dotenv()
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

GLOBAL_MODEL = Gemini(model="gemini-2.0-flash") # Use stable alias

# --- PART 1: GLOBAL TOOL DEFINITIONS ---

def closure_function(context, aggregated_data: str) -> str:
    """Closes the current conversation branch and structures data."""
    return f"CLOSED DATA: {aggregated_data}"

def fulfillment_function(context, closed_data: str) -> str:
    """Generates the final human-readable response."""
    return f"FINAL FULFILLMENT: Plan based on: {closed_data}"

# --- PART 2: GLOBAL AGENT DEFINITIONS (The Logic Components) ---

# Routing Agent (The MCP Target)
routing_agent = LlmAgent(
    model=GLOBAL_MODEL, instruction="You find the best routing options.", tools=[], name="RoutingAgent"
)
# All other agents use simple tools and are defined globally
flight_agent = LlmAgent(
    model=GLOBAL_MODEL, instruction="You find the best flight options.", tools=[google_search], name="FlightAgent"
)
accommodation_agent = LlmAgent(
    model=GLOBAL_MODEL, instruction="You find hotel and accommodation information.", tools=[google_search], name="AccommodationAgent"
)
advisory_agent = LlmAgent(
    model=GLOBAL_MODEL, instruction="You provide weather and travel advisories.", tools=[google_search], name="AdvisoryAgent"
)
closure_agent = LlmAgent(
    model=GLOBAL_MODEL, instruction="You format data from parallel branches.", tools=[closure_function], name="ClosureAgent"
)
fulfillment_agent = LlmAgent(
    model=GLOBAL_MODEL, instruction="You generate final user output.", tools=[fulfillment_function], name="FulfillmentAgent"
)

# The PRIMARY ORCHESTRATOR (Guest Manager Logic)
guest_manager_agent_logic = LlmAgent(
    model=GLOBAL_MODEL,
    name="GuestManagerAgent",
    instruction="You orchestrate the entire travel plan. First call the execute_workflow tool.",
    tools=[] # Tools are injected in the Host's __init__
)


# --- PART 3: GLOBAL WORKFLOW HELPER FUNCTIONS ---

async def travel_workflow_tool(context, user_input: str) -> str:
    """Sequential: Routing Agent -> Flight Agent (Uses global agents)."""
    print("  [TRAVEL WORKFLOW] Starting route & flight search...")
    
    # 1. Routing Agent (Uses MCP tools loaded in set_up)
    route_result = await routing_agent.run_async(f"Route for: {user_input}")
    
    # 2. Flight Agent (Uses Google Search)
    flight_result = await flight_agent.run_async(f"Flights for: {user_input} based on: {route_result.text}")
    
    return f"Route: {route_result.text} | Flights: {flight_result.text}"

async def full_workflow_tool(context, user_input: str) -> str:
    """Orchestrator: Runs Travel, Accomodation, Advisory in parallel, then chains to Closure and Fulfillment."""
    
    print("\n  [ORCHESTRATOR] Starting Parallel Branches...")
    
    # Execute Travel Workflow, Accommodation, and Advisory in parallel
    parallel_tasks = [
        # 1. Travel Workflow (Sequential within its tool)
        travel_workflow_tool(context, user_input),
        
        # 2. Accommodation Agent
        accommodation_agent.run_async(f"Accommodation search for: {user_input}"),
        
        # 3. Advisory Agent
        advisory_agent.run_async(f"Advisory for: {user_input}"),
    ]
    
    # Run all three main branches simultaneously
    results = await asyncio.gather(*parallel_tasks)
    aggregated_data = "\n".join([r.text if hasattr(r, 'text') else str(r) for r in results])
    
    # --- Sequential Output Chain ---
    print("  [ORCHESTRATOR] Starting Sequential Closure Chain...")
    
    # 1. Closure Agent
    closure_output = await closure_agent.run_async(aggregated_data)
    
    # 2. Fulfillment Agent
    final_output = await fulfillment_agent.run_async(closure_output.text)
    
    return final_output.text 


# --- PART 4: THE APPLICATION HOST CLASS (The Technical Bridge) ---

class AdkDeploymentHost(LlmAgent):
    
    # Declare attributes to satisfy Pydantic validation
    mcp_client: McpToolset | None = None 

    def __init__(self):
        """Initializes the Host, inheriting the configuration of the Guest Manager Agent."""
        
        # CRUCIAL FIX: Call super().__init__ FIRST with the orchestrator's config.
        super().__init__(
            model=guest_manager_agent_logic.model,
            name=guest_manager_agent_logic.name,
            instruction=guest_manager_agent_logic.instruction,
            # Inject the full workflow tool into the Host's tool list
            tools=[full_workflow_tool] 
        )

    async def set_up(self):
        """
        FRAMEWORK HOOK: Runs once at startup. Loads MCP tools and injects them into
        the globally defined routing_agent.
        """
        maps_key = os.getenv("MAP_API_KEY")
        if maps_key:
            try:
                self.mcp_client = McpToolset(
                    connection_params=StreamableHTTPServerParams(
                        url="https://mapstools.googleapis.com/mcp", headers={"X-Goog-Api-Key": maps_key}
                    )
                )
                mcp_tools = await self.mcp_client.get_tools()
                
                # INJECT TOOLS INTO THE GLOBAL ROUTING AGENT:
                routing_agent.tools.extend(mcp_tools)
                print(f"✅ HOOK COMPLETE: Injected {len(mcp_tools)} MCP tools into RoutingAgent.")
                
            except Exception as e:
                print(f"❌ HOOK FAILED: Could not connect to MCP server: {e}")

    # Execution methods (query, stream_query) are inherited from LlmAgent.

# --- FINAL EXPORT (Required by the Deployment System) ---

# This object is the instance of the application host that handles the setup.
root_agent = AdkDeploymentHost()
