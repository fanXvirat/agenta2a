# registry_server.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- A2A Imports ---
# We need the AgentCard model to validate incoming registrations
from a2a.types import AgentCard

app = FastAPI(title="AgentOS Agent Registry")

# In-memory dictionary to store registered agents.
# Key: Agent DID (e.g., "did:agent:jane")
# Value: The agent's full AgentCard
AGENT_DIRECTORY: dict[str, AgentCard] = {}

class RegistrationRequest(BaseModel):
    agent_card: AgentCard

@app.post("/register")
async def register_agent(request: RegistrationRequest):
    """
    Receives an AgentCard and registers the agent in the directory.
    The DID is extracted from the agent's URL in the card.
    """
    card = request.agent_card
    # The DID is typically part of the agent's URL. We extract it to use as the key.
    # e.g., from "http://localhost:8002/a2a/did:agent:jane", we get "did:agent:jane"
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
    The `:path` converter is important to correctly handle DIDs containing colons.
    """
    print(f"REGISTRY: Received resolution request for DID '{agent_did}'")
    card = AGENT_DIRECTORY.get(agent_did)
    if not card:
        print(f"REGISTRY: Agent with DID '{agent_did}' not found.")
        raise HTTPException(status_code=404, detail=f"Agent with DID '{agent_did}' not found.")
    
    print(f"REGISTRY: Found agent '{agent_did}'. Returning card.")
    # Return the card as a dictionary
    return card.model_dump(mode='json', exclude_none=True)

if __name__ == "__main__":
    print("--- Starting AgentOS Registry Server ---")
    print("Endpoint available at http://127.0.0.1:8004")
    uvicorn.run(app, host="127.0.0.1", port=8004)