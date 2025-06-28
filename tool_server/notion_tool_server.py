# tool_server/notion_agent.py

import uvicorn
import httpx
import json
import traceback
import notion_client

# A2A SDK Imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCard, AgentSkill, TextPart, Part

class NotionAgentExecutor(AgentExecutor):
    def get_api_key_from_context(self, context: RequestContext) -> str:
        """Securely extract the API key from the call context provided by the SDK."""
        headers = context.call_context.state.get("headers", {})
        api_key = headers.get("x-notion-api-key")
        if not api_key:
            raise PermissionError("Notion API key was not provided in the 'X-Notion-API-Key' header.")
        return api_key

    def create_page_in_database(self, database_id: str, title: str, content: str, api_key: str) -> dict:
        try:
            notion = notion_client.Client(auth=api_key)
            print(f"NOTION AGENT: Authenticated to Notion. Creating page titled '{title}'.")
            
            parent = {"database_id": database_id}
            properties = {"title": {"title": [{"text": {"content": title}}]}}
            children = [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]}}]
            
            response = notion.pages.create(parent=parent, properties=properties, children=children)
            print(f"NOTION AGENT: Page created successfully. URL: {response['url']}")
            return {"status": "success", "page_url": response['url']}
        except notion_client.errors.APIResponseError as e:
            print(f"NOTION AGENT: Notion API Error - {e}")
            return {"error": f"Notion API Error: {e.body}"}
        except Exception as e:
            print(f"NOTION AGENT: An unexpected error occurred - {e}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            api_key = self.get_api_key_from_context(context)
            tool_call_str = context.get_user_input()
            tool_call = json.loads(tool_call_str)
            tool_args = tool_call.get("args", {})
            
            result = self.create_page_in_database(api_key=api_key, **tool_args)
            result_str = json.dumps(result)
            
            final_message = task_updater.new_agent_message(parts=[Part(root=TextPart(text=result_str))])
            await task_updater.complete(message=final_message)
        except Exception as e:
            traceback.print_exc()
            error_message = task_updater.new_agent_message(parts=[Part(root=TextPart(text=f"Failed to execute Notion tool: {str(e)}"))])
            await task_updater.failed(message=error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue): pass

# --- Agent Server Setup ---
if __name__ == "__main__":
    AGENT_DID = "did:agent:notion"
    AGENT_URL = f"http://localhost:8005/a2a/{AGENT_DID}"
    REGISTRY_URL = "http://localhost:8004"
    
    agent_card = AgentCard(
        name="Notion Agent",
        description="A specialized worker agent that interacts with the Notion API on behalf of a user.",
        version="1.0", url=AGENT_URL, capabilities={"streaming": False},
        defaultInputModes=["application/json"], defaultOutputModes=["application/json"],
        skills=[AgentSkill(
            id="create_page_in_database", name="Create Notion Page",
            description="Creates a page in a Notion database. Requires 'X-Notion-API-Key' header.",
            tags=["notion", "database"]
        )]
    )
    
    executor = NotionAgentExecutor()
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    async def startup_event():
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{REGISTRY_URL}/register", json={"agent_card": agent_card.model_dump(mode='json')})
            print(f"NOTION AGENT: Successfully registered with DID '{AGENT_DID}'.")
        except Exception as e:
            print(f"NOTION AGENT: CRITICAL - Could not register. Error: {e}")

    app = a2a_app.build(rpc_url=f"/a2a/{AGENT_DID}", on_startup=[startup_event])
    
    print("--- Starting AgentOS Notion Agent ---")
    uvicorn.run(app, host="127.0.0.1", port=8005)