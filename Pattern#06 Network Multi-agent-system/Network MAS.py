"""
Network (Swarm) Multi-Agent Pattern — LangGraph Implementation
=============================================================

This script implements the Network/Peer-to-Peer pattern where:
1. There is NO central supervisor.
2. A Flight Agent handles flights and can transfer to the Hotel Agent.
3. A Hotel Agent handles hotels and can transfer to the Flight Agent.
4. Agents communicate directly by utilizing "transfer tools" to route the graph.

Architecture:
    START → Flight Agent ↔ Hotel Agent
                  ↓            ↓
                Tools        Tools

Key Concepts:
- Decentralized routing: Agents decide when to hand off control.
- Transfer Tools: Dummy tools used specifically to signal graph routing.
- Context Sharing: The messages list is the shared ground truth.
"""

# ==============================================================================
# Step 1: Imports and Setup
# ==============================================================================

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# We use the same model for all peers. Note: For tool calling reliability, 
# a strong model is usually recommended.
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)


# ==============================================================================
# Step 2: Define Tools and Transfer Mechanisms
# ==============================================================================

# ── Domain Tools ──────────────────────────────────────────

@tool
def search_flights(destination: str) -> str:
    """Mock tool to search for flights."""
    return f"Flights to {destination}: Flight AA123 ($450), Flight DL456 ($500)."

@tool
def search_hotels(destination: str) -> str:
    """Mock tool to search for hotels."""
    return f"Hotels in {destination}: Grand Hotel ($200/night), Budget Inn ($80/night)."

# ── Transfer Tools (The Magic of Network MAS) ─────────────

@tool
def transfer_to_flight_agent():
    """Call this tool to transfer the conversation to the Flight Agent."""
    # The return string here is what the LLM sees after calling the tool.
    return "Transferred to Flight Agent."

@tool
def transfer_to_hotel_agent():
    """Call this tool to transfer the conversation to the Hotel Agent."""
    return "Transferred to Hotel Agent."


# ==============================================================================
# Step 3: Define the Agents
# ==============================================================================

# Create specific toolsets for each agent. Notice each gets the ability to 
# transfer to the OTHER agent.
flight_tools = [search_flights, transfer_to_hotel_agent]
hotel_tools = [search_hotels, transfer_to_flight_agent]

# Bind tools to models
flight_model = model.bind_tools(flight_tools)
hotel_model = model.bind_tools(hotel_tools)


def flight_agent_node(state: MessagesState) -> dict:
    """The Flight Expert."""
    messages = state["messages"]
    
    # System prompt injects their persona and instructions on when to transfer
    sys_msg = SystemMessage(content=(
        "You are a Flight Expert. You help users find flights. "
        "If the user also needs hotel information or mentions hotels, you MUST use the "
        "`transfer_to_hotel_agent` tool to pass control to the Hotel Expert. "
        "Do NOT try to answer hotel questions yourself. "
        "If you have answered the flight portion and transferred to the hotel agent, "
        "or if the user only wanted flights, you are done."
    ))
    
    response = flight_model.invoke([sys_msg] + messages)
    
    # We optionally tag the message so we can trace who said what later
    response.name = "flight_agent"
    
    return {"messages": [response]}


def hotel_agent_node(state: MessagesState) -> dict:
    """The Hotel Expert."""
    messages = state["messages"]
    
    sys_msg = SystemMessage(content=(
        "You are a Hotel Expert. You help users find hotels. "
        "If the user also needs flight information or mentions flights, you MUST use the "
        "`transfer_to_flight_agent` tool to pass control to the Flight Expert. "
        "Do NOT try to answer flight questions yourself. "
        "If you have answered the hotel portion and transferred to the flight agent, "
        "or if the user only wanted hotels, you are done."
    ))
    
    response = hotel_model.invoke([sys_msg] + messages)
    response.name = "hotel_agent"
    
    return {"messages": [response]}


# ==============================================================================
# Step 4: The Routing Logic (Conditional Edges)
# ==============================================================================

# Standard tool nodes to execute the actual Python functions
flight_tool_node = ToolNode(flight_tools)
hotel_tool_node = ToolNode(hotel_tools)


def flight_router(state: MessagesState) -> str:
    """Routes after the flight agent speaks."""
    last_msg = state["messages"][-1]
    
    if not last_msg.tool_calls:
        # No tool called -> Agent gave final text answer
        return END
        
    # Check WHICH tool was called
    tool_name = last_msg.tool_calls[0]["name"]
    if tool_name == "transfer_to_hotel_agent":
        return "hotel_agent"
    
    # Otherwise, it's a standard tool like search_flights
    return "flight_tools"


def hotel_router(state: MessagesState) -> str:
    """Routes after the hotel agent speaks."""
    last_msg = state["messages"][-1]
    
    if not last_msg.tool_calls:
        return END
        
    tool_name = last_msg.tool_calls[0]["name"]
    if tool_name == "transfer_to_flight_agent":
        return "flight_agent"
    
    return "hotel_tools"


# ==============================================================================
# Step 5: Build the Graph
# ==============================================================================

graph = StateGraph(MessagesState)

# Add Nodes
graph.add_node("flight_agent", flight_agent_node)
graph.add_node("hotel_agent", hotel_agent_node)
graph.add_node("flight_tools", flight_tool_node)
graph.add_node("hotel_tools", hotel_tool_node)

# Add Edges
# Since we don't have a supervisor, we need a default entry point.
graph.add_edge(START, "flight_agent") 

# Tools always return to the agent that called them
graph.add_edge("flight_tools", "flight_agent")
graph.add_edge("hotel_tools", "hotel_agent")

# Add Routing (Peer-to-Peer Handoffs)
graph.add_conditional_edges(
    "flight_agent", 
    flight_router,
    {"hotel_agent": "hotel_agent", "flight_tools": "flight_tools", END: END}
)

graph.add_conditional_edges(
    "hotel_agent", 
    hotel_router,
    {"flight_agent": "flight_agent", "hotel_tools": "hotel_tools", END: END}
)

workflow = graph.compile()


# ==============================================================================
# Step 6: Run It
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌐 NETWORK (SWARM) MAS — Starting")
    print("=" * 60)

    # The user asks for both flights and hotels
    request = "I want to go to Paris. Can you find me a flight and a hotel?"
    
    print(f"\nUser: {request}")
    
    result = workflow.invoke({"messages": [HumanMessage(content=request)]})
    
    print("\n" + "=" * 60)
    print("🎯 FINAL ANSWER")
    print("=" * 60)
    print(result["messages"][-1].content)
    
    print("\n" + "=" * 60)
    print("🔄 NETWORK HANDOFF TRACE")
    print("=" * 60)
    
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            agent_name = getattr(msg, "name", "Unknown Agent")
            
            # If the agent called a tool
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc['name']
                    if "transfer" in tool_name:
                        print(f"[{agent_name.upper()}] ➡️  HANDOFF: {tool_name}()")
                    else:
                        print(f"[{agent_name.upper()}] 🛠️  TOOL USED: {tool_name}()")
            
            # If the agent spoke text
            elif msg.content:
                print(f"[{agent_name.upper()}] 💬  SPEAKS: {msg.content[:100]}...")
