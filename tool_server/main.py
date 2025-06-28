# tool_server/general_tools_agent.py

import uvicorn
import httpx
import json

# A2A SDK Imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCard, AgentSkill, TextPart, Part

# Tooling Imports
from duckduckgo_search import DDGS

# --- Tool Implementation ---

MOCK_CALENDAR = {
    "testuser": {"today": "10 AM: Standup", "tomorrow": "2 PM: Project brainstorming"},
    "jane": {"today": "Morning: Yoga", "tomorrow": "1 PM: Lunch with Alex"}
}

def web_search(query: str) -> str:
    print(f"GENERAL_TOOLS: Performing web search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        return json.dumps(results) if results else "No results found."
    except Exception as e:
        return json.dumps({"error": f"Failed to perform web search: {e}"})

def check_calendar(user: str, date: str) -> str:
    print(f"GENERAL_TOOLS: Checking calendar for user '{user}' on date '{date}'")
    user_calendar = MOCK_CALENDAR.get(user.lower(), {})
    lower_date = date.lower()
    search_key = "today" if "today" in lower_date else "tomorrow"
    events = user_calendar.get(search_key, "You have nothing scheduled.")
    return f"On {date}, your schedule is: {events}"

# --- A2A AgentExecutor Implementation ---

class GeneralToolsExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            tool_call_str = context.get_user_input()
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call.get("tool_name")
            tool_args = tool_call.get("args", {})
            
            result_str = ""
            if tool_name == "web_search":
                result_str = web_search(**tool_args)
            elif tool_name == "check_calendar":
                result_str = check_calendar(**tool_args)
            else:
                result_str = f"Error: Tool '{tool_name}' not supported by this agent."
                
            final_message = task_updater.new_agent_message(parts=[Part(root=TextPart(text=result_str))])
            await task_updater.complete(message=final_message)

        except Exception as e:
            error_message = task_updater.new_agent_message(parts=[Part(root=TextPart(text=f"Tool execution failed: {e}"))])
            await task_updater.failed(message=error_message)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass

# --- Agent Server Setup ---
if __name__ == "__main__":
    AGENT_DID = "did:agent:general_tools"
    AGENT_URL = f"http://localhost:8003/a2a/{AGENT_DID}"
    REGISTRY_URL = "http://localhost:8004"
    
    agent_card = AgentCard(
        name="General Tools Agent",
        description="A worker agent providing web search and calendar checking tools.",
        version="1.0",
        url=AGENT_URL,
        capabilities={"streaming": False},
        defaultInputModes=["application/json"],
        defaultOutputModes=["application/json"],
        skills=[
            AgentSkill(id="web_search", name="Web Search", description="Performs a web search.", tags=["search"]),
            AgentSkill(id="check_calendar", name="Check Calendar", description="Checks a user's mock calendar.", tags=["calendar"])
        ]
    )
    
    executor = GeneralToolsExecutor()
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    async def startup_event():
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{REGISTRY_URL}/register", json={"agent_card": agent_card.model_dump(mode='json')})
            print(f"GENERAL_TOOLS: Successfully registered with DID '{AGENT_DID}'.")
        except Exception as e:
            print(f"GENERAL_TOOLS: CRITICAL - Could not register with directory. Error: {e}")

    app = a2a_app.build(rpc_url=f"/a2a/{AGENT_DID}", on_startup=[startup_event])
    
    print("--- Starting General Tools Agent Server ---")
    uvicorn.run(app, host="127.0.0.1", port=8003)