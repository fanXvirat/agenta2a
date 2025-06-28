# agent_os/platform_ui.py

import uvicorn
import httpx
import uuid
import json
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from a2a.client import A2AClient, create_text_message_object
from a2a.types import (
    SendMessageRequest, MessageSendParams, AgentCard, Message, Task, 
    SendMessageSuccessResponse, JSONRPCErrorResponse
)
from a2a.utils.message import get_message_text

app = FastAPI(title="AgentOS Platform UI")
templates = Jinja2Templates(directory="templates")

REGISTRY_URL = "http://localhost:8004"
CURRENT_USER_DID = "did:agent:testuser"

class IntentRequest(BaseModel):
    intent: str
    
class TaskResponseRequest(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the main user interface."""
    return templates.TemplateResponse("index.html", context={"request": request})

@app.post("/api/execute_intent")
async def execute_intent(intent_request: IntentRequest):
    """
    Takes an intent from the user, finds their personal agent, and makes a 
    blocking A2A call to it.
    """
    print(f"UI SERVER: Received intent: '{intent_request.intent}'")
    
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{REGISTRY_URL}/resolve/{CURRENT_USER_DID}")
            res.raise_for_status()
            agent_card = AgentCard.model_validate(res.json())
        
        print(f"UI SERVER: Found agent '{agent_card.name}'. Sending intent...")

        async with httpx.AsyncClient() as a2a_http_client:
            client = A2AClient(httpx_client=a2a_http_client, agent_card=agent_card)
            message = create_text_message_object(content=intent_request.intent)
            
            # *** THIS IS THE FIX ***
            # The `acceptedOutputModes` field is required by the MessageSendConfiguration model.
            config = {
                "blocking": True,
                "acceptedOutputModes": ["text/plain", "application/json"] 
            }
            params = MessageSendParams(message=message, configuration=config)
            a2a_req = SendMessageRequest(id=str(uuid.uuid4()), params=params)
            
            response_model = await client.send_message(request=a2a_req, http_kwargs={'timeout': 180.0})

        print(f"UI SERVER: Received final response from agent.")

        if isinstance(response_model.root, SendMessageSuccessResponse):
            result = response_model.root.result
            final_text = "Task completed, but no final message was returned."
            if isinstance(result, Message):
                final_text = get_message_text(result)
            elif isinstance(result, Task) and result.status.message:
                final_text = get_message_text(result.status.message)
            return JSONResponse(content={"response": final_text})
        
        elif isinstance(response_model.root, JSONRPCErrorResponse):
            error_data = response_model.root.error.model_dump()
            return JSONResponse(status_code=500, content={"error": error_data.get("message", "An unknown agent error occurred.")})
        
        else:
            return JSONResponse(status_code=500, content={"error": "Received an unknown response type from the agent."})
             
    except httpx.HTTPStatusError as e:
        error_text = e.response.text
        try:
            error_json = json.loads(error_text)
            return JSONResponse(status_code=e.response.status_code, content={"error": error_json})
        except json.JSONDecodeError:
            return JSONResponse(status_code=e.response.status_code, content={"error": error_text})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred in the UI server: {str(e)}"})

@app.get("/api/my_pending_tasks")
async def get_my_pending_tasks():
    print(f"UI SERVER: Fetching pending tasks for {CURRENT_USER_DID}")
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{REGISTRY_URL}/resolve/{CURRENT_USER_DID}")
            res.raise_for_status()
            agent_card = AgentCard.model_validate(res.json())
            
            response = await client.post(
                agent_card.url,
                json={
                    "jsonrpc": "2.0",
                    "method": "agent/getPendingTasks",
                    "params": {},
                    "id": str(uuid.uuid4())
                },
                timeout=30.0
            )
            response.raise_for_status()
            return JSONResponse(response.json())
            
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to get pending tasks: {str(e)}"})

@app.post("/api/tasks/{task_id}/respond")
async def respond_to_task(task_id: str, response_request: TaskResponseRequest):
    print(f"UI SERVER: Sending response for task {task_id}: '{response_request.response}'")
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{REGISTRY_URL}/resolve/{CURRENT_USER_DID}")
            res.raise_for_status()
            agent_card = AgentCard.model_validate(res.json())

            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            message = create_text_message_object(content=response_request.response)
            message.taskId = task_id 
            
            # *** THIS IS THE FIX ***
            # The `acceptedOutputModes` field is required here as well.
            config = {
                "blocking": True,
                "acceptedOutputModes": ["text/plain", "application/json"]
            }
            params = MessageSendParams(message=message, configuration=config)
            a2a_req = SendMessageRequest(id=str(uuid.uuid4()), params=params)

            response_model = await a2a_client.send_message(request=a2a_req, http_kwargs={'timeout': 180.0})
        
        return JSONResponse({"status": "response sent", "agent_reply": response_model.model_dump(mode='json')})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to respond to task: {str(e)}"})


if __name__ == "__main__":
    print("--- Starting AgentOS UI Server ---")
    uvicorn.run(app, host="127.0.0.1", port=8002)