# AgentA2A: A Decentralized Autonomous Agent Ecosystem

This project implements a decentralized ecosystem where autonomous AI agents can discover, negotiate with, and delegate tasks to one another. Using **Google's A2A Protocol**, this framework moves beyond centralized orchestrators, allowing for a true peer-to-peer network of collaborative agents.

---

## 🏛️ Architectural Vision: A Peer-to-Peer Agent Network

The architecture is designed as a distributed network of sovereign agents. There is no central point of failure or control. Discovery, communication, and negotiation are all handled directly between agents via the A2A Protocol.

```
                                << A2A Protocol >>
                               /                  \
+-------------------------+   /                    \   +-------------------------+
|  Your Personal Agent    |<-------------------------->|  Jane's Personal Agent  |
|  (Planner w/ Gemini)    |                            |  (Planner w/ Gemini)    |
|                         |   \                    /   |                         |
|  - Google Calendar Tool |    \                  /    |  - Slack Tool           |
|  - Notion Tool (Private)|     \                /     |  - Other Tools...       |
+-------------------------+      \              /      +-------------------------+
       ^                          \            /
       |                           \          /
       | (User Interface)           `--------`
       |                           /          \
+------|------------------+       /            \       +-------------------------+
|      v                  |      /              \      | Public Notion Service   |
|   Web Dashboard         |<---->  << A2A Protocol >>  <| Agent (3rd Party)       |
|   (UI/UX for Your Agent)|      \                      |                         |
+-------------------------+       \                     | - create_page           |
                                   \                    | - query_database        |
                                    \                   +-------------------------+
```

### The Workflow: An Example

1.  **User Prompt:** A user asks their Personal Agent: *"Check if Jane is free tomorrow at 2 PM, and if so, book a meeting and create a Notion page for it."*
2.  **Internal Planning:** The user's agent, using **Google Gemini**, breaks this down:
    *   `Step 1: Discover and query Jane's agent for availability.`
    *   `Step 2: If available, confirm the meeting.`
    *   `Step 3: Discover and delegate page creation to a Notion agent.`
3.  **Agent-to-Agent Discovery:** The Personal Agent uses the **A2A Protocol** to broadcast a discovery request for an agent owned by "Jane".
4.  **Inter-Agent Communication:** Jane's Personal Agent responds. The two agents negotiate a query for the calendar. Jane's agent runs its internal calendar tool and returns her availability, all without exposing the underlying tool or calendar details.
5.  **Service Agent Interaction:** After confirming, the user's agent discovers a public, third-party **Notion Service Agent** via the A2A protocol. It securely delegates the task of creating the meeting page.
6.  **Final Confirmation:** The Personal Agent confirms all tasks are complete and reports back to the user.

---

## ✨ An Open Agent Ecosystem

This architecture enables a true marketplace of agents.
*   **Personal Agents:** Users have their own sovereign agents that manage their private tools and data.
*   **Service Agents:** Companies can deploy their own specialized agents (e.g., a "Slack Agent," "Salesforce Agent") that other agents can discover and use. This creates a powerful, decentralized platform for services.

## Key Technologies

*   **A2A Protocol:** The lifeblood of the system. This protocol handles:
    *   **Agent Discovery:** Finding other agents on the network without a central directory.
    *   **Secure Communication:** End-to-end encrypted messaging between agents.
    *   **Task Negotiation:** A standardized way for agents to offer, accept, and reject tasks.
*   **Google Gemini:** The advanced language model used by agents for planning and function-calling.
*   **FastAPI:** Powers the web dashboard for user interaction with their Personal Agent.

## 🚀 Running the System Locally

This is a multi-service system. Each agent and UI is an independent process.

### Prerequisites
*   Python 3.10+
*   `pip` and `virtualenv`

### Installation & Setup
1.  **Clone the repo and set up the environment**
    ```sh
    git clone https://github.com/fanXvirat/agenta2a.git
    cd agent-a2a
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Set up environment variables**
    *   Create a `.env` file and add your `GEMINI_API_KEY`, `NOTION_API_KEY`, etc.

### Launching the Services

Each component must be run in a separate terminal.

1.  **Start the Tool/Service Agents**
    ```sh
    # In a new terminal (e.g., a public Notion agent)
    python tool_server/notion_tool_server.py
    ```
2.  **Start the Personal Agents**
    *   Each user agent runs as a separate instance on a different port.
    ```sh
    # In a new terminal for your agent
    python agent_os/personal_agent.py --user testuser --port 8000

    # In another terminal for Jane's agent
    python agent_os/personal_agent.py --user jane --port 8001
    ```
3.  **Start the Main UI**
    ```sh
    # In a final terminal
    python agent_os/platform_main.py
    ```
4.  Access your agent's dashboard at `http://localhost:8002`.
