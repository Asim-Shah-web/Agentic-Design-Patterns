"""
ReAct (Reason + Act) Pattern — LangGraph Implementation
=====================================================

This script implements the internal logic of the ReAct pattern from scratch.
Instead of using the prebuilt `create_react_agent`, we manually build the 
StateGraph to demonstrate the loop between reasoning and acting.

Architecture:
    START -> Reasoner (LLM) <---> Executor (Tools)
                 |
                 v
                END
"""

import os
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# ==============================================================================
# Step 1: Define Tools
# ==============================================================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    if location.lower() == "london":
        return "It is currently 15°C and raining in London."
    elif location.lower() == "tokyo":
        return "It is currently 22°C and sunny in Tokyo."
    return f"It is mild and 20°C in {location}."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use for math problems."""
    try:
        # Note: eval is used here strictly for the tutorial mock environment.
        # NEVER use raw eval() in production code.
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"

tools = [get_weather, calculate]
tool_mapping = {tool.name: tool for tool in tools}

# Setup Model and Bind Tools
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model_with_tools = model.bind_tools(tools)


# ==============================================================================
# Step 2: The Reasoner Node (Thought/Action Generation)
# ==============================================================================

def reasoner_node(state: MessagesState):
    print("\n🧠 [REASONER]: Thinking...")
    
    # The LLM reads the history and responds
    response = model_with_tools.invoke(state["messages"])
    
    # If the LLM makes a tool call, we can print its "Thought" process
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"   -> Thought: I need to use '{tc['name']}' with args: {tc['args']}")
    else:
        print("   -> Thought: I have enough information to answer.")
        
    # We append the LLM's response to the state
    return {"messages": [response]}


# ==============================================================================
# Step 3: The Executor Node (Observation Generation)
# ==============================================================================

def executor_node(state: MessagesState):
    print("\n🛠️ [EXECUTOR]: Executing tools...")
    
    # The last message in the state is the AI's tool call
    last_message = state["messages"][-1]
    
    # We must return a list of ToolMessages containing the observations
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        # Find the actual python function
        tool_function = tool_mapping[tool_call["name"]]
        
        # Execute the function
        print(f"   -> Running {tool_call['name']}...")
        observation = tool_function.invoke(tool_call["args"])
        print(f"   -> Observation: {observation}")
        
        # Create the ToolMessage (this acts as the 'Observation' sent back to LLM)
        tool_messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )
        )
        
    # Appending the tool messages back to the state
    return {"messages": tool_messages}


# ==============================================================================
# Step 4: The Conditional Router
# ==============================================================================

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    
    # If there are tool calls, we must execute them
    if getattr(last_message, "tool_calls", None):
        return "continue"
    # Otherwise, we are done
    return "end"


# ==============================================================================
# Step 5: Compile the Graph
# ==============================================================================

workflow = StateGraph(MessagesState)

# Add Nodes
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("executor", executor_node)

# Add Edges
workflow.add_edge(START, "reasoner")

# The conditional edge decides if we loop or stop
workflow.add_conditional_edges(
    "reasoner",
    should_continue,
    {
        "continue": "executor",
        "end": END
    }
)

# After tools execute, ALWAYS go back to the reasoner to observe!
workflow.add_edge("executor", "reasoner")

react_graph = workflow.compile()


# ==============================================================================
# Step 6: Run It
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔄 REACT PATTERN — Internal Logic Demo")
    print("=" * 60)

    # We ask a multi-step question that forces the ReAct loop to cycle multiple times
    request = "What is the weather in London? Also, what is 45 * 12?"
    print(f"User Request: {request}\n")

    # Run the workflow
    result = react_graph.invoke({"messages": [HumanMessage(content=request)]})

    print("\n" + "=" * 60)
    print("🎯 FINAL OUTPUT TO USER")
    print("=" * 60)
    
    # The last message is the final answer from the Reasoner
    print(result["messages"][-1].content)
    print("\nDone.")
