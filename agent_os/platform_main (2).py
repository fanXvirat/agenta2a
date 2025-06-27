# platform_main.py (Final version with Asynchronous Delegation & Polling)

import os
import uvicorn
import httpx
import uuid
import asyncio
import traceback
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from cryptography.fernet import Fernet, InvalidToken

# --- SDK & Google Imports ---

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.client import A2AClient, create_text_message_object
from a2a.types import (
    AgentCard, AgentSkill, Message, MessageSendParams, SendMessageRequest,
    SendMessageSuccessResponse, TextPart, TaskState, TaskStatus, Role, Task
)
from a2a.utils.task import new_task

from google import genai
from google.genai import types as genai_types
from starlette.concurrency import run_in_threadpool
from duckduckgo_search import DDGS
class IntentRequest(BaseModel):
    intent: str
# --- 1. SETUP and CONFIGURATION ---
app = FastAPI(title="AgentOS Platform")
templates = Jinja2Templates(directory="templates")
ENCRYPTION_KEY = b'h3aA-pmPhY1gwysxT84a51-vWqVprC4mG15MvBplHXA='
cipher_suite = Fernet(ENCRYPTION_KEY)
REGISTRY_URL = "http://localhost:8004"
TASK_MANAGER_URL = "http://localhost:8006"

# --- 2. DATA MODELS & STATE ---
TEMP_USER_DB = {}
AGENT_REGISTRY = {}
CURRENT_USER = None

class PlannerState:
    def __init__(self, task_id: str, llm_history: list[dict]):
        self.task_id = task_id
        self.llm_history = llm_history

# --- 3. SECURITY & USER MANAGEMENT (Unchanged) ---
def encrypt_key(api_key: str) -> bytes: return cipher_suite.encrypt(api_key.encode())
def decrypt_key(encrypted_key: bytes) -> str:
    try: return cipher_suite.decrypt(encrypted_key).decode()
    except InvalidToken: print("ERROR: Decryption failed."); raise

async def _register_agent_with_directory(card: AgentCard):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{REGISTRY_URL}/register", json={"agent_card": card.model_dump(mode='json')})
            print(f"PLATFORM: Successfully registered DID {card.url.split('/')[-1]} with the registry.")
    except Exception as e: print(f"PLATFORM ERROR: Could not register agent with the directory. {e}")

async def register_user(username: str, gemini_api_key: str, notion_api_key: str | None = None):
    if username in TEMP_USER_DB: raise ValueError("User already exists")
    user_data = {"encrypted_gemini_api_key": encrypt_key(gemini_api_key)}
    if notion_api_key: user_data["encrypted_notion_api_key"] = encrypt_key(notion_api_key)
    TEMP_USER_DB[username] = user_data
    agent_did = f"did:agent:{username.lower()}"
    agent_url = f"http://localhost:8002/a2a/{agent_did}"
    capabilities = { "skills": ["general assistance", "conversation", "planning"], "tools": ["web_search", "calendar_management", "notion"]}
    agent_card = AgentCard(
        name=f"{username.capitalize()}'s Personal Agent",
        description=f"An autonomous agent representing {username} with skills in: {', '.join(capabilities['skills'])}.",
        version="2.0", url=agent_url, capabilities={"streaming": False},
        defaultInputModes=["text/plain"], defaultOutputModes=["text/plain"],
        skills=[AgentSkill(id=skill.replace(' ', '_'), name=skill.title(), description=f"Can provide {skill}.", tags=[skill]) for skill in capabilities['skills']]
    )
    AGENT_REGISTRY[agent_did] = {"card": agent_card, "username": username, "capabilities": capabilities}
    await _register_agent_with_directory(agent_card)
    print(f"USER_MGMT: Registered '{username}' with DID '{agent_did}'.")

async def login_user(username: str):
    global CURRENT_USER
    if username not in TEMP_USER_DB: raise ValueError("User not found")
    CURRENT_USER = username
    print(f"USER_MGMT: User '{username}' logged in.")

# --- 4. A2A & PLANNER ---
class PublicAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue): pass
    async def cancel(self, context: RequestContext, event_queue: EventQueue): pass

async def find_agent_by_username(username: str) -> dict | None:
    did_to_find = f"did:agent:{username.lower()}" if not username.startswith("did:agent:") else username
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{REGISTRY_URL}/resolve/{did_to_find}")
            if response.status_code == 404: return None
            response.raise_for_status()
            agent_card_data = response.json()
            return {"card": AgentCard.model_validate(agent_card_data), "username": username}
    except Exception as e:
        print(f"PLATFORM ERROR: Could not resolve agent. {e}")
        return None

async def run_conversational_planner(planner_state: PlannerState, task: Task, api_key: str, agent_user: str) -> str:
    genai_client = genai.Client(api_key=api_key)
    
    # --- UPDATED TOOL DESCRIPTION ---
    ask_tool = { "name": "ask_another_agent", "description": "Asks a question to another user's agent and waits for their response. Use this for tasks that require delegation.", "parameters": {"type": "object", "properties": {"username": {"type": "string"}, "question": {"type": "string"}}, "required": ["username", "question"]}}
    add_to_calendar_tool = {"name": "add_to_calendar", "description": "Adds an event to the user's calendar.", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "description": {"type": "string"}}, "required": ["date", "time", "description"]}}
    
    notion_tool = {"name": "create_page_in_database", "description": "Creates a page in a Notion database. DELEGATES to the 'notion' agent.", "parameters": {"type": "object", "properties": {"database_id": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"}},"required": ["database_id", "title", "content"]}}
    finish_tool = {"name": "finish_and_summarize", "description": "Call this ONLY when the user's goal is fully achieved. Provide a complete, final summary.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},"required": ["summary"]}}
    search_tool = {"name": "web_search", "description": "Performs a web search.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
    calendar_tool = {"name": "check_calendar", "description": "Checks the user's calendar for availability on a specific date.", "parameters": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}}
    tools_config = genai_types.Tool(function_declarations=[ask_tool, search_tool, calendar_tool, notion_tool, finish_tool,add_to_calendar_tool])
    generate_config = genai_types.GenerateContentConfig(tools=[tools_config])
    
    model_request_contents = planner_state.llm_history
    if len(model_request_contents) == 1:
        original_goal = task.history[0].parts[0].root.text
        system_prompt = f"""
You are a master planner. Your job is to create and execute a step-by-step plan to achieve the user's high-level goal. The user's original goal is: "{original_goal}". Review the conversation history and the original goal. Execute the *next single step* required to make progress. If the goal is complex, you will need to call multiple tools in sequence across several turns. Do NOT call `finish_and_summarize` until all parts of the original goal have been fully completed. The user's first message is: {original_goal}"""
        model_request_contents[0]['parts'][0]['text'] = system_prompt

    for turn in range(7):
        print(f"\n--- CONVERSATION TURN {turn + 1}/7 ({agent_user}) ---")
        print(f"AGENT [{agent_user}]: Planning next action...")
        response = await run_in_threadpool(genai_client.models.generate_content, model='gemini-2.5-flash-lite-preview-06-17', contents=model_request_contents, config=generate_config)
        
        try:
            candidate = response.candidates[0]
            if not (candidate.content.parts and candidate.content.parts[0].function_call):
                final_text = candidate.content.parts[0].text if (candidate.content.parts and candidate.content.parts[0].text) else "Planner did not choose a tool."
                return final_text
            function_call = candidate.content.parts[0].function_call
        except Exception as e: return f"Planner returned an invalid response structure: {e}"

        model_request_contents.append(candidate.content)
        tool_name, tool_args = function_call.name, dict(function_call.args)
        print(f"AGENT [{agent_user}]: Planner chose tool: '{tool_name}' with args: {tool_args}")

        if tool_name == "finish_and_summarize": return tool_args.get('summary', "Done.")
        
        tool_result = ""
        # --- REFACTORED 'ask_another_agent' FOR ASYNC POLLING ---
        if tool_name == "ask_another_agent":
            target_username, question = tool_args.get('username'), tool_args.get('question')
            if not target_username or not question: tool_result = "Error: username and question are required."
            else:
                target_agent_info = await find_agent_by_username(target_username)
                if not target_agent_info: tool_result = f"Error: Could not find agent '{target_username}'."
                else:
                    try:
                        print(f"AGENT [{agent_user}]: Delegating task to '{target_username}' and waiting...")
                        async with httpx.AsyncClient() as http_client:
                            a2a_request = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams(message=create_text_message_object(content=question)))
                            response = await http_client.post(target_agent_info['card'].url, json=a2a_request.model_dump(mode='json'), timeout=30.0)
                            response.raise_for_status()
                            initial_response_data = response.json()
                            subtask_id = initial_response_data.get("task_id")

                            if not subtask_id:
                                tool_result = "Error: Delegated agent did not return a task_id."
                            else:
                                print(f"AGENT [{agent_user}]: Sub-task '{subtask_id}' created. Now polling for result...")
                                for i in range(10): 
                                    await asyncio.sleep(3)
                                    print(f"  - Polling attempt {i+1}/10 for task {subtask_id}...")
                                    task_response = await http_client.get(f"{TASK_MANAGER_URL}/tasks/{subtask_id}")
                                    task_data = task_response.json()
                                    if task_data.get("status", {}).get("state") == "completed":
                                        tool_result = task_data.get("result", "Task completed without a result.")
                                        print(f"AGENT [{agent_user}]: Sub-task {subtask_id} complete. Result: {tool_result}")
                                        break
                                else:
                                    tool_result = f"Error: Task {subtask_id} timed out."
                    except Exception as e: tool_result = f"Failed to communicate with {target_username}'s agent. Error: {e}"
        elif tool_name == "create_page_in_database":
            notion_key = TEMP_USER_DB.get(agent_user, {}).get("encrypted_notion_api_key")
            if not notion_key: tool_result = "Error: Notion API key not found for this user."
            else:
                try:
                    decrypted_notion_key = decrypt_key(notion_key)
                    notion_agent_info = await find_agent_by_username("notion")
                    if not notion_agent_info: tool_result = "Error: The Notion Agent is not registered."
                    else:
                        tool_call_for_worker = {"tool_name": "create_page_in_database", "args": tool_args}
                        message_to_notion_agent = create_text_message_object(content=json.dumps(tool_call_for_worker))
                        headers = {"X-Notion-API-Key": decrypted_notion_key}
                        a2a_request_payload = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams(message=message_to_notion_agent))
                        async with httpx.AsyncClient() as http_client:
                            response = await http_client.post(notion_agent_info['card'].url, json=a2a_request_payload.model_dump(mode='json'), headers=headers, timeout=30.0)
                            response.raise_for_status()
                            response_from_agent = SendMessageSuccessResponse.model_validate(response.json())
                        if isinstance(response_from_agent.result, Message): tool_result = response_from_agent.result.parts[0].root.text
                        else: tool_result = "Invalid response from notion agent."
                except Exception as e: tool_result = f"Error during Notion page creation: {e}"
        elif tool_name == "check_calendar":
            date = tool_args.get('date', 'today')
            try:
                async with httpx.AsyncClient() as http_client:
                    tool_payload = {"tool_name": "check_calendar", "args": {"user": agent_user, "date": date}}
                    tool_response = await http_client.post("http://localhost:8003/execute_tool", json=tool_payload, timeout=30.0)
                    tool_response.raise_for_status()
                tool_result = str(tool_response.json().get('result', 'No result found.'))
            except Exception as e: tool_result = f"Error during calendar check: {e}"
        elif tool_name == "web_search":
            query = tool_args.get('query', '')
            if not query: tool_result = "Error: query is required."
            else:
                try:
                    async with httpx.AsyncClient() as http_client:
                        tool_response = await http_client.post("http://localhost:8003/execute_tool", json={"tool_name": "web_search", "args": tool_args}, timeout=30.0)
                        tool_response.raise_for_status()
                    tool_result = str(tool_response.json().get('result', 'No result found.'))
                except Exception as e: tool_result = f"Error during web search: {e}"
        else: tool_result = f"Error: Planner chose an unhandled tool '{tool_name}'."
        
        model_request_contents.append({'role': 'user', 'parts': [{'function_response': {'name': tool_name, 'response': {'content': tool_result}}}]})
        
        history_entry = f"Tool Used: {tool_name}, Args: {tool_args}, Result: {tool_result}"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{TASK_MANAGER_URL}/tasks/{task.id}/update", json={"history_entry": history_entry}, timeout=5.0)
        except Exception as e: print(f"PLATFORM ERROR: Could not update task {task.id} in Task Manager. {e}")
            
    return "Conversation finished due to reaching max turns."

# --- 6. API ENDPOINTS & MAIN BLOCK ---
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request): return templates.TemplateResponse(name="index.html", context={"request": request})

@app.post("/api/execute_intent")
async def execute_intent(request: IntentRequest):
    global CURRENT_USER
    if not CURRENT_USER: raise HTTPException(status_code=401, detail="User not logged in.")
    try:
        decrypted_key = decrypt_key(TEMP_USER_DB[CURRENT_USER]["encrypted_gemini_api_key"])
        initial_message = create_text_message_object(content=request.intent, role=Role.user)
        task = None
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{TASK_MANAGER_URL}/tasks", json=initial_message.model_dump(mode='json'), timeout=5.0)
            response.raise_for_status()
            task = Task.model_validate(response.json())
        if not task: raise HTTPException(status_code=500, detail="Task object could not be created.")
        
        initial_llm_history = [{'role': 'user', 'parts': [{'text': request.intent}]}]
        planner_state = PlannerState(task_id=task.id, llm_history=initial_llm_history)
        
        final_answer = await run_conversational_planner(planner_state, task, decrypted_key, CURRENT_USER)
        return JSONResponse(content={"response": final_answer})
    except Exception as e: 
        print(f"API ERROR: {e}"); 
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- NEW: Background Task Runner for A2A Handler ---
async def run_and_update_task_in_background(planner_state: PlannerState, task: Task, api_key: str, agent_username: str):
    """Runs the planner and updates the task manager with the final result."""
    print(f"BACKGROUND TASK: Starting planner for task {task.id} for agent {agent_username}.")
    final_answer = await run_conversational_planner(planner_state, task, api_key, agent_username)
    
    print(f"BACKGROUND TASK: Planner for task {task.id} finished. Updating task manager with result.")
    update_payload = {"status": "completed", "result": final_answer}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{TASK_MANAGER_URL}/tasks/{task.id}/update", json=update_payload)
    except Exception as e:
        print(f"BACKGROUND TASK ERROR: Could not update task {task.id} with final result. Error: {e}")

@app.post("/a2a/{agent_did:path}")
async def handle_a2a_request(agent_did: str, request: Request, background_tasks: BackgroundTasks):
    if agent_did not in AGENT_REGISTRY: raise HTTPException(status_code=404, detail="Agent DID not found.")
    agent_username = AGENT_REGISTRY[agent_did]["username"]
    print(f"A2A HANDLER: Handling request for agent {agent_did} (user: {agent_username})")
    try:
        body = await request.json()
        a2a_request = SendMessageRequest.model_validate(body)
        incoming_message = a2a_request.params.message
        
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{TASK_MANAGER_URL}/tasks", json=incoming_message.model_dump(mode='json'), timeout=5.0)
            response.raise_for_status()
            task = Task.model_validate(response.json())
        if not task: raise Exception("Task object could not be created by A2A handler.")

        decrypted_key = decrypt_key(TEMP_USER_DB[agent_username]["encrypted_gemini_api_key"])
        initial_llm_history = [{'role': 'user', 'parts': [{'text': incoming_message.parts[0].root.text }]}]
        planner_state = PlannerState(task_id=task.id, llm_history=initial_llm_history)
        
        background_tasks.add_task(run_and_update_task_in_background, planner_state, task, decrypted_key, agent_username)
        
        print(f"A2A HANDLER: Task {task.id} created and started in background. Returning task ID to caller.")
        return JSONResponse(content={"status": "accepted", "task_id": task.id}, status_code=202)
    except Exception as e:
        print(f"A2A HANDLER ERROR: {e}"); traceback.print_exc()
        request_id = locals().get('a2a_request', {}).id if 'a2a_request' in locals() else "unknown"
        error_response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": f"Internal error: {str(e)}"}}
        return JSONResponse(content=error_response, status_code=500)

async def main():
    print("--- Server Starting: Simulating User Setup ---")
    load_dotenv()
    my_api_key = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    notion_api_key = os.getenv("NOTION_API_KEY", "YOUR_NOTION_API_KEY_HERE")
    if "YOUR_GEMINI_API_KEY_HERE" in my_api_key:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); print("!!! WARNING: Gemini API Key not found.                     !!!"); print("!!! Please set your GEMINI_API_KEY environment variable.   !!!"); print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if "YOUR_NOTION_API_KEY_HERE" in notion_api_key:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); print("!!! WARNING: Notion API Key not found.                     !!!"); print("!!! You will not be able to use Notion tools.              !!!"); print("!!! Please set your NOTION_API_KEY environment variable.   !!!"); print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    await register_user("testuser", my_api_key, notion_api_key)
    jane_api_key = os.getenv("JANE_GEMINI_API_KEY", my_api_key)
    await register_user("jane", jane_api_key, None)
    await login_user("testuser")
    print("--- User Setup Complete. Starting Platform Server ---")
    print("Dashboard available at http://127.0.0.1:8002")
    config = uvicorn.Config(app, host="127.0.0.1", port=8002, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())