# task_manager_service.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# A2A Imports
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import Task, Message, TaskState
from a2a.client import create_text_message_object

app = FastAPI(title="AgentOS Task Manager")

# This service uses the same TaskStore primitive. In a production system,
# this would be backed by a persistent database (e.g., Redis, PostgreSQL).
TASK_STORE = InMemoryTaskStore()

# --- Pydantic Models ---

class TaskUpdateRequest(BaseModel):
    """Defines the shape for a request to update a task."""
    status: Optional[TaskState] = None
    history_entry: Optional[str] = None
    # NEW: A field to store the final result of a task upon completion.
    result: Optional[str] = None

# We extend the base A2A Task model to include our custom 'result' field.
# This allows us to store the final outcome of a task separately from its status.
class TaskWithResult(Task):
    """An extended Task model that includes a final result field."""
    result: Optional[str] = Field(default=None)


# --- API Endpoints ---

@app.post("/tasks", response_model=TaskWithResult, status_code=201)
async def create_task(initial_message: Message):
    """
    Creates a new task based on an initial message and stores it.
    """
    # Use the same new_task utility from the A2A SDK
    from a2a.utils.task import new_task
    task_data = new_task(request=initial_message).model_dump()
    # Create the task using our extended model to ensure the 'result' field exists
    task = TaskWithResult(**task_data)
    await TASK_STORE.save(task)
    print(f"TASK MANAGER: Created new task with ID: {task.id}")
    return task

@app.get("/tasks/{task_id}", response_model=TaskWithResult)
async def get_task(task_id: str):
    """
    Retrieves the current state of a task.
    """
    print(f"TASK MANAGER: Retrieving task with ID: {task_id}")
    task = await TASK_STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Ensure the returned task conforms to TaskWithResult for consistency
    if not isinstance(task, TaskWithResult):
        task = TaskWithResult(**task.model_dump())
    return task

@app.post("/tasks/{task_id}/update", response_model=TaskWithResult)
async def update_task(task_id: str, update_request: TaskUpdateRequest):
    """
    Updates the status, history, or final result of a task.
    """
    print(f"TASK MANAGER: Updating task with ID: {task_id}")
    task = await TASK_STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Ensure we are working with the model that has the result field
    if not isinstance(task, TaskWithResult):
        task = TaskWithResult(**task.model_dump())

    if update_request.status:
        print(f"  - Setting status to: {update_request.status.value}")
        task.status.state = update_request.status
    
    if update_request.history_entry:
        print(f"  - Adding history entry: {update_request.history_entry[:70]}...")
        if task.history is None:
            task.history = []
        history_message = create_text_message_object(
            content=update_request.history_entry,
            role="agent" 
        )
        task.history.append(history_message)

    # NEW: Save the final result to the task object
    if update_request.result is not None:
        print(f"  - Setting final result: {update_request.result[:70]}...")
        task.result = update_request.result

    await TASK_STORE.save(task)
    return task

if __name__ == "__main__":
    print("--- Starting AgentOS Task Manager Service ---")
    print("Endpoint available at http://127.0.0.1:8006")
    uvicorn.run(app, host="127.0.0.1", port=8006)