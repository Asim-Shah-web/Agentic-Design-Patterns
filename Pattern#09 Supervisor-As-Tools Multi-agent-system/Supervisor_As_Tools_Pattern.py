"""
Supervisor as Tools Pattern — LangGraph Implementation
======================================================

This script implements the "Supervisor as Tools" (Sub-agents as Tools) pattern where:
1. Specialists (Researcher, Writer) are wrapped as standard LangChain tools.
2. The main Supervisor agent is a tool-calling agent that delegates work to these specialists via tool invocation.
3. The supervisor receives the output of the specialists as tool messages, keeping its own context window clean and the graph architecture simple.

Architecture:
               ┌───────> [Tools Node] ────────┐
    START ──> Supervisor Agent <───────────────┴── (Loop until done) ──> END
                 │
                 ├── Calls Research Agent (as a Tool)
                 └── Calls Writing Agent (as a Tool)
"""

# ==============================================================================
# Step 1: Imports and Setup
# ==============================================================================

import operator
import os
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# Bind models
# The Supervisor uses a capable, reasoning model (GPT-4o)
supervisor_model = ChatOpenAI(model="gpt-4o", temperature=0)

# Specialists use a faster, cost-efficient model (GPT-4o-mini)
specialist_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize Search Tool
search_tool = TavilySearchResults(max_results=3)


# ==============================================================================
# Step 2: Define the Specialized Agents (Sub-Agents)
# ==============================================================================

def run_research_agent(query: str) -> str:
    """
    Sub-agent that conducts web search and synthesizes findings.
    
    This agent runs in its own isolated scope, preventing search raw data 
    from bleeding into the supervisor's main context history.
    """
    print(f"🔍 [Research Agent] Researching query: '{query}'...")
    search_results = search_tool.invoke(query)
    
    # Format the results for the specialist model
    context = "\n".join([f"- {r.get('content')} (Source: {r.get('url')})" for r in search_results])
    
    prompt = f"""You are an expert research analyst. Your job is to answer the query below using the provided search results:
Query: {query}

Search Results:
{context}

Provide a factual, bulleted list summarizing the key data and findings."""
    
    response = specialist_model.invoke([HumanMessage(content=prompt)])
    return response.content


def run_writing_agent(notes: str, format_instructions: str) -> str:
    """
    Sub-agent that structures raw notes into polished markdown.
    """
    print(f"✍️  [Writing Agent] Formatting and organizing notes...")
    
    prompt = f"""You are a professional editor. Take the unstructured research notes below 
and format them into a polished markdown report according to the formatting instructions.

Notes:
{notes}

Formatting Instructions:
{format_instructions}"""
    
    response = specialist_model.invoke([HumanMessage(content=prompt)])
    return response.content


# ==============================================================================
# Step 3: Wrap Sub-Agents as Tools
# ==============================================================================

@tool
def research_sub_agent_tool(query: str) -> str:
    """Delegates a research task to a specialized Research Agent. 
    Use this to look up current events, stock performance, statistics, or general facts on the web."""
    return run_research_agent(query)


@tool
def writing_sub_agent_tool(notes: str, format_instructions: str) -> str:
    """Delegates formatting and editing tasks to a specialized Writing Agent.
    Use this tool when you have gathered raw research notes and need to organize 
    them into a cohesive, structured markdown document."""
    return run_writing_agent(notes, format_instructions)


# Define tool library
tools = [research_sub_agent_tool, writing_sub_agent_tool]
tool_node = ToolNode(tools)


# ==============================================================================
# Step 4: Define the Graph State
# ==============================================================================

class SupervisorState(TypedDict):
    """
    State representing the entire conversation flow for the supervisor.
    
    Fields:
        messages: The sequence of messages containing supervisor and tool nodes outputs.
    """
    messages: Annotated[List[BaseMessage], operator.add]


# ==============================================================================
# Step 5: Define the Supervisor Node & Routing Logic
# ==============================================================================

# Bind the specialist tools to the supervisor LLM
supervisor_llm_with_tools = supervisor_model.bind_tools(tools)

def supervisor_agent(state: SupervisorState) -> dict:
    """
    The main coordinator node. Evaluates user input and decides whether 
    to call one of the specialist tools, or respond directly if the task is complete.
    """
    print(f"\n🧠 [Supervisor] Evaluating current state...")
    messages = state["messages"]
    
    system_prompt = SystemMessage(content="""You are an expert research supervisor.
Your job is to orchestrate the generation of high-quality research reports. 
You do not perform search or formatting yourself. Instead, you delegate tasks to your specialists:
1. Call 'research_sub_agent_tool' when you need to gather facts or look up current events.
2. Call 'writing_sub_agent_tool' when you have raw research findings and need to compile them into a polished markdown report.

Guidance:
- Only delegate one logical step at a time.
- Read the tool responses carefully. Once you have the final formatted report from the Writing Agent, present the final output to the user and stop.
- Do not run tools in infinite loops. If you have all the necessary information, output the final result.""")
    
    # Prepend system prompt to the chat history
    response = supervisor_llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


def should_continue(state: SupervisorState) -> str:
    """
    Conditional routing edge to determine if the LLM wants to call a tool, 
    or if it has completed its work.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # Route to the ToolNode to run the sub-agent tool
        print(f"🔗 [Supervisor] Delegating to Sub-Agent tool: '{last_message.tool_calls[0]['name']}'")
        return "tools"
    # Otherwise, stop and return to the user
    print("🏁 [Supervisor] Task complete. Returning final answer.")
    return END


# ==============================================================================
# Step 6: Build and Compile the Graph
# ==============================================================================

builder = StateGraph(SupervisorState)

# Add Nodes
builder.add_node("supervisor", supervisor_agent)
builder.add_node("tools", tool_node)

# Add Edges
builder.add_edge(START, "supervisor")

# Conditional edge after supervisor node execution
builder.add_conditional_edges(
    "supervisor",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# After running tools, always loop back to the supervisor
builder.add_edge("tools", "supervisor")

# Compile
workflow = builder.compile()


# ==============================================================================
# Step 7: Run the Supervisor
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 SUPERVISOR AS TOOLS PIPELINE — Starting")
    print("=" * 60)
    
    # Query requiring sequential research and formatting
    user_query = (
        "Write a structured 2-paragraph report comparing the current stock performance "
        "and market capitalizations of Apple and Microsoft in 2026. Use the research sub-agent "
        "to gather data, and the writing sub-agent to format it professionally."
    )
    
    result = workflow.invoke({
        "messages": [HumanMessage(content=user_query)]
    })
    
    print("\n" + "=" * 80)
    print("🎯 FINAL CONVERSATION OUTPUT")
    print("=" * 80)
    
    # Find the final text message in the conversation history
    for msg in reversed(result["messages"]):
        if msg.content and not hasattr(msg, "tool_calls") and msg.__class__.__name__ != "ToolMessage":
            print(f"[{msg.__class__.__name__}]:\n\n{msg.content}\n")
            break
            
    print("=" * 80)
