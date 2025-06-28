# agent_os/personal_agent.py

import os
import uvicorn
import httpx
import uuid
import asyncio
import traceback
import json
import argparse
from dotenv import load_dotenv

# SDK & Google Imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.client import A2AClient, create_text_message_object
from a2a.types import (
    AgentCard, AgentSkill, Message, MessageSendParams, SendMessageRequest,
    SendMessageSuccessResponse, Part, TextPart, Task, TaskState, JSONRPCErrorResponse
)
from a2a.utils.message import get_message_text
from a2a.utils.errors import ServerError
from starlette.requests import Request
from starlette.responses import JSONResponse

from google import genai
from google.genai import types as genai_types
from starlette.concurrency import run_in_threadpool


# --- CONFIGURATION ---
REGISTRY_URL = "http://localhost:8004"
NOTION_AGENT_DID = "did:agent:notion"
GENERAL_TOOLS_AGENT_DID = "did:agent:general_tools"


# --- Custom Request Handler ---
class PersonalAgentRequestHandler(DefaultRequestHandler):
    async def on_get_pending_tasks(self, params: dict = None, context: dict = None) -> list[Task]:
        executor_username = getattr(self.agent_executor, 'username', 'Unknown')
        print(f"PERSONAL_AGENT ({executor_username}): Received request to get pending tasks.")
        if not isinstance(self.task_store, InMemoryTaskStore): return []
        
        return [
            task for task in self.task_store.tasks.values() 
            if task.status.state == TaskState.input_required
        ]


# --- The Agent's "Brain" ---
class PersonalAgentExecutor(AgentExecutor):
    def __init__(self, username: str, gemini_api_key: str, notion_api_key: str | None):
        self.username = username
        self.notion_api_key = notion_api_key
        self.gemini_api_key = gemini_api_key

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            original_goal = get_message_text(context.message)
            await task_updater.start_work(message=task_updater.new_agent_message(parts=[Part(root=TextPart(text=f"OK, planning how to achieve: \"{original_goal}\""))]))
            
            # The planner function is now a method of the executor.
            final_answer = await self.run_conversational_planner(context, task_updater)
            
            await task_updater.complete(message=task_updater.new_agent_message(parts=[Part(root=TextPart(text=final_answer))]))

        except Exception as e:
            traceback.print_exc()
            await task_updater.failed(message=task_updater.new_agent_message(parts=[Part(root=TextPart(text=f"Planner failed: {str(e)}"))]))

    async def run_conversational_planner(self, context: RequestContext, task_updater: TaskUpdater) -> str:
        # This logic is now directly adapted from your working prototype.
        genai_client = genai.Client(api_key=self.gemini_api_key)
        
        ask_tool = { "name": "ask_another_agent", "description": "Asks a question to another user's agent.", "parameters": {"type": "object", "properties": {"username": {"type": "string"}, "question": {"type": "string"}}, "required": ["username", "question"]}}
        notion_tool = {"name": "create_page_in_database", "description": "Creates a page in a Notion database.", "parameters": {"type": "object", "properties": {"database_id": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"}},"required": ["database_id", "title", "content"]}}
        finish_tool = {"name": "finish_and_summarize", "description": "Call this ONLY when the user's goal is fully achieved.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},"required": ["summary"]}}
        search_tool = {"name": "web_search", "description": "Performs a web search.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
        calendar_tool = {"name": "check_calendar", "description": "Checks the user's calendar for availability.", "parameters": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}}
        
        tools_config = genai_types.Tool(function_declarations=[ask_tool, search_tool, calendar_tool, notion_tool, finish_tool])
        generate_config = genai_types.GenerateContentConfig(tools=[tools_config])
        
        original_goal = get_message_text(context.message)
        system_prompt = f"You are a master planner. Your job is to create and execute a step-by-step plan to achieve the user's high-level goal: \"{original_goal}\". Execute the *next single step*. Do NOT call `finish_and_summarize` until the goal is fully achieved."
        
        model_request_contents = [
            {'role': 'user', 'parts': [{'text': system_prompt + "\n\nUser's Goal: " + original_goal}]}
        ]

        for turn in range(7):
            print(f"\n--- CONVERSATION TURN {turn + 1}/7 ({self.username}) ---")
            response = await run_in_threadpool(genai_client.models.generate_content, model='gemini-2.5-flash-lite-preview-06-17', contents=model_request_contents, config=generate_config)
            
            try:
                candidate = response.candidates[0]
                if not (candidate.content.parts and candidate.content.parts[0].function_call):
                    # CORRECTED: Access .text from the parent response object
                    return response.text or "Planner did not choose a tool or provide text."
                function_call = candidate.content.parts[0].function_call
            except Exception as e:
                return f"Planner returned an invalid response structure: {e}"

            model_request_contents.append(candidate.content)
            tool_name, tool_args = function_call.name, dict(function_call.args)
            await task_updater.update_status(TaskState.working, message=task_updater.new_agent_message(parts=[Part(root=TextPart(text=f"Requesting tool: `{tool_name}`"))]))

            if tool_name == "finish_and_summarize":
                return tool_args.get('summary', "Done.")
            
            tool_result = ""
            if tool_name == "ask_another_agent":
                tool_result = await self._a2a_call_user_agent(tool_args['username'], tool_args['question'])
            elif tool_name == "create_page_in_database":
                tool_result = await self._a2a_call_notion_agent(tool_args)
            elif tool_name == "check_calendar":
                tool_result = await self._a2a_call_general_tools(tool_name, {"date": tool_args.get("date")})
            elif tool_name == "web_search":
                 tool_result = await self._a2a_call_general_tools(tool_name, {"query": tool_args.get("query")})
            else:
                tool_result = f"Error: Planner chose an unhandled tool '{tool_name}'."
            
            model_request_contents.append({'role': 'user', 'parts': [{'function_response': {'name': tool_name, 'response': {'content': tool_result}}}]})
            
        return "Conversation finished due to reaching max turns."

    async def _resolve_and_call_agent(self, agent_did: str, message_content: str, headers: dict = None) -> str:
        """A generic helper to resolve an agent and send it a blocking A2A message."""
        async with httpx.AsyncClient() as client:
            try:
                reg_res = await client.get(f"{REGISTRY_URL}/resolve/{agent_did}")
                if reg_res.status_code == 404: return f"Error: Agent with DID '{agent_did}' not found in registry."
                reg_res.raise_for_status()
                card = AgentCard.model_validate(reg_res.json())
                
                a2a_client = A2AClient(httpx_client=client, agent_card=card)
                message = create_text_message_object(content=message_content)
                params = MessageSendParams(message=message, configuration={"blocking": True, "acceptedOutputModes": ["text/plain", "application/json"]})
                a2a_req = SendMessageRequest(id=str(uuid.uuid4()), params=params)
                response_model = await a2a_client.send_message(request=a2a_req, http_kwargs={'headers': headers} if headers else None)
                
                if isinstance(response_model.root, SendMessageSuccessResponse):
                    result = response_model.root.result
                    if isinstance(result, Message): return get_message_text(result)
                    if isinstance(result, Task) and result.status.message: return get_message_text(result.status.message)
                    return f"Delegation to {agent_did} finished with status: {result.status.state.value}"
                
                if isinstance(response_model.root, JSONRPCErrorResponse):
                    return f"Agent {agent_did} returned an error: {response_model.root.error.message}"
                return "Error: Agent returned an unexpected payload."
            except Exception as e: return f"A2A communication failed for {agent_did}: {traceback.format_exc()}"

    async def _a2a_call_notion_agent(self, args: dict) -> str:
        if not self.notion_api_key: return "Error: Notion API key is not configured for this user."
        headers = {"X-Notion-API-Key": self.notion_api_key}
        payload = json.dumps({"tool_name": "create_page_in_database", "args": args})
        return await self._resolve_and_call_agent(NOTION_AGENT_DID, payload, headers)

    async def _a2a_call_general_tools(self, tool_name: str, args: dict) -> str:
        if not tool_name: return "Error: general tool name not provided."
        payload = json.dumps({"tool_name": tool_name, "args": {**args, "user": self.username}})
        return await self._resolve_and_call_agent(GENERAL_TOOLS_AGENT_DID, payload)

    async def _a2a_call_user_agent(self, username: str, question: str) -> str:
        return await self._resolve_and_call_agent(f"did:agent:{username.lower()}", question)
        
    async def cancel(self, context: RequestContext, event_queue: EventQueue): pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Personal Agent Server.")
    parser.add_argument("--user", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    cli_args = parser.parse_args()

    load_dotenv()
    gemini_key_env_var = f"{cli_args.user.upper()}_GEMINI_API_KEY"
    gemini_api_key = os.getenv(gemini_key_env_var, os.getenv("GEMINI_API_KEY"))
    notion_api_key = os.getenv("NOTION_API_KEY") if cli_args.user == 'testuser' else None

    if not gemini_api_key: raise ValueError(f"API Key for {cli_args.user} not found in env.")

    agent_did = f"did:agent:{cli_args.user.lower()}"
    agent_card = AgentCard(
        name=f"{cli_args.user.capitalize()}'s Personal Agent", description="An autonomous agent.",
        version="2.0", url=f"http://localhost:{cli_args.port}/a2a/{agent_did}",
        capabilities={"streaming": False}, defaultInputModes=["text/plain"], defaultOutputModes=["text/plain"],
        skills=[AgentSkill(id="planning", name="Planning", description="Can execute multi-step plans.", tags=["planning"])]
    )

    executor = PersonalAgentExecutor(username=cli_args.user, gemini_api_key=gemini_api_key, notion_api_key=notion_api_key)
    handler = PersonalAgentRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    original_handler = a2a_app._handle_requests
    async def new_a2a_dispatcher(request: Request):
        body_bytes = await request.body()
        async def receive_clone(): return {'type': 'http.request', 'body': body_bytes}
        new_request = Request(request.scope, receive_clone)
        
        try:
            body_json = json.loads(body_bytes)
            if body_json.get("method") == "agent/getPendingTasks":
                pending_tasks = await handler.on_get_pending_tasks()
                return JSONResponse([task.model_dump(mode='json') for task in pending_tasks])
        except (json.JSONDecodeError, KeyError): pass
        
        return await original_handler(new_request)

    a2a_app._handle_requests = new_a2a_dispatcher

    async def startup_event():
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{REGISTRY_URL}/register", json={"agent_card": agent_card.model_dump(mode='json', by_alias=True)})
            print(f"PERSONAL AGENT ({cli_args.user}): Successfully registered.")
        except Exception as e:
            print(f"PERSONAL AGENT ({cli_args.user}): CRITICAL - Could not register. Error: {e}")

    from starlette.applications import Starlette
    from starlette.routing import Route
    
    app = Starlette(on_startup=[startup_event])
    app.add_route(f"/a2a/{agent_did}", new_a2a_dispatcher, methods=["POST"])
    app.add_route('/.well-known/agent.json', a2a_app._handle_get_agent_card, methods=['GET'])
    
    print(f"--- Starting Personal Agent Server for '{cli_args.user}' on port {cli_args.port} ---")
    uvicorn.run(app, host="127.0.0.1", port=cli_args.port, log_level="info")