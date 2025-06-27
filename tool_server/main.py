import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from duckduckgo_search import DDGS

app = FastAPI(title="AgentOS Tool Server")

class ToolExecutionRequest(BaseModel):
    tool_name: str
    args: dict

class ToolExecutionResponse(BaseModel):
    status: str
    result: str | list | dict

MOCK_CALENDAR = {
    "testuser": {
        "today": "10 AM: Standup. 2 PM: Dentist appointment.",
        "tomorrow": "All day: Working on the AgentOS project.",
    },
    "jane": {
        "today": "Morning: Yoga. Afternoon: Write project proposal.",
        "tomorrow": "9 AM: Team Sync. 1 PM: Lunch with Alex.",
    }
}

def web_search(query: str) -> list[dict]:
    # ... (this function remains the same)
    print(f"TOOL SERVER: Performing web search for query: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        print(f"TOOL SERVER: Found {len(results)} results.")
        return results
    except Exception as e:
        return [{"error": f"Failed to perform web search: {e}"}]

def check_calendar(user: str, date: str) -> str:
    """
    Checks a user's mock calendar for a specific date.
    NOW with smarter date interpretation.
    """
    print(f"TOOL SERVER: Checking calendar for user '{user}' on date: '{date}'")
    user_calendar = MOCK_CALENDAR.get(user.lower(), {})
    
    # --- NEW: Smarter Date Logic ---
    lower_date = date.lower()
    search_key = lower_date
    
    # Handle relative terms by mapping them to our mock data keys
    if "today" in lower_date or "tonight" in lower_date or "afternoon" in lower_date or "evening" in lower_date:
        search_key = "today"
    elif "tomorrow" in lower_date:
        search_key = "tomorrow"
        
    events = user_calendar.get(search_key, f"You have nothing scheduled on {date}.")
    return f"On {date}, your schedule is: {events}"

def add_to_calendar(user: str, date: str, time: str, description: str) -> str:
    # ... (this function remains the same)
    print(f"TOOL SERVER: Adding event for user '{user}' on {date} at {time}: '{description}'")
    if user not in MOCK_CALENDAR: MOCK_CALENDAR[user] = {}
    new_event = f"{time}: {description}"
    if date in MOCK_CALENDAR[user]: MOCK_CALENDAR[user][date] += f". {new_event}"
    else: MOCK_CALENDAR[user][date] = new_event
    return f"Success: The event '{description}' has been added to {user}'s calendar for {date} at {time}."

@app.post("/execute_tool", response_model=ToolExecutionResponse)
async def execute_tool(request: ToolExecutionRequest):
    # ... (this function remains the same, no changes needed here)
    print(f"TOOL SERVER: Received request to execute tool: '{request.tool_name}' with args: {request.args}")
    if request.tool_name == "web_search":
        if "query" not in request.args: raise HTTPException(status_code=400, detail="Missing 'query'")
        return ToolExecutionResponse(status="success", result=web_search(query=request.args["query"]))
    elif request.tool_name == "check_calendar":
        if "date" not in request.args or "user" not in request.args: raise HTTPException(status_code=400, detail="Missing 'date' or 'user'")
        return ToolExecutionResponse(status="success", result=check_calendar(user=request.args["user"], date=request.args["date"]))
    elif request.tool_name == "add_to_calendar":
        required_args = ["user", "date", "time", "description"]
        if not all(k in request.args for k in required_args): raise HTTPException(status_code=400, detail=f"Missing args. Required: {required_args}")
        return ToolExecutionResponse(status="success", result=add_to_calendar(**request.args))
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found.")

if __name__ == "__main__":
    print("--- Starting AgentOS Tool Server ---")
    print("Available Tools: [web_search, check_calendar, add_to_calendar]")
    uvicorn.run(app, host="127.0.0.1", port=8003)