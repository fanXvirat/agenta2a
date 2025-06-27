# notion_agent.py

import uvicorn
import httpx
import json
import os
import traceback
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
# --- NEW: Import the Notion Client ---
import notion_client

# A2A Imports
from a2a.types import AgentCard, AgentSkill, Message, SendMessageSuccessResponse, SendMessageRequest, Role
from a2a.client import create_text_message_object

app = FastAPI(title="AgentOS Notion Agent")

# --- Agent Identity ---
NOTION_AGENT_DID = "did:agent:notion"
NOTION_AGENT_URL = f"http://localhost:8005/a2a/{NOTION_AGENT_DID}"
REGISTRY_URL = "http://localhost:8004"

# --- 1. Implement the REAL Tool Logic ---
def create_page_in_database(database_id: str, title: str, content: str, api_key: str) -> dict:
    """
    Creates a new page in a Notion database using the actual Notion API.
    """
    print(f"NOTION AGENT: Received request to create page titled '{title}' in DB '{database_id}'.")
    if not api_key or "YOUR_NOTION_API_KEY" in api_key:
        print("NOTION AGENT: ERROR - Missing or invalid API Key.")
        return {"error": "Authentication error: Notion API key is missing or invalid."}
    
    try:
        # Initialize the Notion client with the user-specific API key
        notion = notion_client.Client(auth=api_key)
        
        print("NOTION AGENT: Authentication successful. Creating page...")
        
        # Define the structure for the new page
        parent = {"database_id": database_id}
        properties = {
            "title": {
                "title": [{"text": {"content": title}}]
            }
        }
        children = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                }
            }
        ]
        
        # Make the API call to create the page
        response = notion.pages.create(parent=parent, properties=properties, children=children)
        
        print(f"NOTION AGENT: Successfully created page. URL: {response['url']}")
        
        return {
            "status": "success",
            "page_url": response['url']
        }
    except notion_client.errors.APIResponseError as e:
        print(f"NOTION AGENT: Notion API Error - {e}")
        return {"error": f"Notion API Error: {e}"}
    except Exception as e:
        print(f"NOTION AGENT: An unexpected error occurred - {e}")
        return {"error": f"An unexpected error occurred: {e}"}

# ... (The rest of the file remains the same) ...

# --- 2. Create the A2A Endpoint ---
@app.post("/a2a/{agent_did:path}")
async def handle_a2a_request(agent_did: str, request: Request):
    if agent_did != NOTION_AGENT_DID:
        raise HTTPException(status_code=404, detail="Agent DID not found.")
    print(f"NOTION AGENT: Received A2A request for {agent_did}")
    api_key = request.headers.get("X-Notion-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-Notion-API-Key header is required.")
    a2a_request = None
    try:
        body = await request.json()
        a2a_request = SendMessageRequest.model_validate(body)
        tool_call_str = a2a_request.params.message.parts[0].root.text
        tool_call = json.loads(tool_call_str)
        tool_name = tool_call.get("tool_name")
        tool_args = tool_call.get("args", {})
        if tool_name == "create_page_in_database":
            result = create_page_in_database(
                database_id=tool_args.get("database_id"),
                title=tool_args.get("title"),
                content=tool_args.get("content"),
                api_key=api_key
            )
            final_answer = json.dumps(result)
        else:
            final_answer = json.dumps({"error": f"Tool '{tool_name}' not supported by this agent."})
        reply_message = create_text_message_object(role=Role.agent, content=final_answer)
        response_payload = SendMessageSuccessResponse(id=a2a_request.id, result=reply_message)
        return response_payload.model_dump(mode='json', exclude_none=True)
    except Exception as e:
        print(f"NOTION AGENT ERROR: {e}"); traceback.print_exc()
        request_id = a2a_request.id if a2a_request else "unknown"
        error_response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": str(e)}}
        return JSONResponse(content=error_response, status_code=500)

# --- 3. Agent Self-Registration on Startup ---
@app.on_event("startup")
async def startup_event():
    notion_skill = AgentSkill(
        id="create_page_in_database",
        name="Create Notion Page",
        description="Creates a new page with specified content and title in a Notion database. Expects a JSON input with 'database_id', 'title', and 'content'.",
        tags=["notion", "database", "writing"]
    )
    agent_card = AgentCard(
        name="Notion Agent",
        description="A specialized worker agent that interacts with the Notion API.",
        version="1.0",
        url=NOTION_AGENT_URL,
        capabilities={"streaming": False},
        defaultInputModes=["application/json"],
        defaultOutputModes=["application/json"],
        skills=[notion_skill]
    )
    print("NOTION AGENT: Attempting to register with the registry...")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{REGISTRY_URL}/register", json={"agent_card": agent_card.model_dump(mode='json')}, timeout=5.0)
            print(f"NOTION AGENT: Successfully registered with DID '{NOTION_AGENT_DID}'.")
    except Exception as e:
        print(f"NOTION AGENT: CRITICAL - Could not register with the directory. Is the registry server running at {REGISTRY_URL}? Error: {e}")

if __name__ == "__main__":
    print("--- Starting AgentOS Notion Agent ---")
    print(f"This agent will register itself as '{NOTION_AGENT_DID}'")
    uvicorn.run(app, host="127.0.0.1", port=8005)