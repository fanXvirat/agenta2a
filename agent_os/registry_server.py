# agent_os/registry_server.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- A2A Imports ---
from a2a.types import AgentCard

app = FastAPI(title="AgentOS Agent Registry")

# In-memory dictionary to store registered agents.
AGENT_DIRECTORY: dict[str, AgentCard] = {}

class RegistrationRequest(BaseModel):
    agent_card: AgentCard

@app.post("/register")
async def register_agent(request: RegistrationRequest):
    """
    Receives an AgentCard and registers the agent in the directory.
    The DID is extracted from the agent's URL.
    """
    card = request.agent_card
    did = card.url.split('/')[-1]
    
    if not did.startswith("did:agent:"):
        raise HTTPException(status_code=400, detail="Invalid agent URL format in AgentCard, cannot extract DID.")
        
    print(f"REGISTRY: Registering agent with DID '{did}'")
    AGENT_DIRECTORY[did] = card
    return {"status": "success", "did": did}

@app.get("/resolve/{agent_did:path}")
async def resolve_agent(agent_did: str):
    """
    Resolves an agent's DID to their AgentCard.
    The `:path` converter handles DIDs with colons.
    """
    print(f"REGISTRY: Received resolution request for DID '{agent_did}'")
    card = AGENT_DIRECTORY.get(agent_did)
    if not card:
        print(f"REGISTRY: Agent with DID '{agent_did}' not found.")
        raise HTTPException(status_code=404, detail=f"Agent with DID '{agent_did}' not found.")
    
    print(f"REGISTRY: Found agent '{agent_did}'. Returning card.")
    return card.model_dump(mode='json', by_alias=True, exclude_none=True)

if __name__ == "__main__":
    print("--- Starting AgentOS Registry Server ---")
    uvicorn.run(app, host="127.0.0.1", port=8004)