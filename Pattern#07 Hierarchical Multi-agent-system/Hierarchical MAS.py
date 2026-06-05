"""
Hierarchical Multi-Agent Pattern — LangGraph Implementation
===========================================================

This script implements a Nested Supervision (Hierarchical) pattern:
1. CEO (Top-Level Supervisor) delegates to two departments:
2. Research Department (Subgraph):
   - Research Supervisor
   - Web Researcher Agent
   - Wiki Researcher Agent
3. Writing Department (Subgraph):
   - Writing Supervisor
   - Copywriter Agent
   - Editor Agent

Architecture:
    CEO -> Research Team (Subgraph) -> CEO -> Writing Team (Subgraph) -> CEO -> FINISH

Key Concepts:
- Subgraphs: Entire teams are compiled as independent StateGraphs and used as nodes in the top graph.
- Information Hiding: The CEO only sees the final output of the team, not the internal chatter.
- Scalability: You can add dozens of teams without overwhelming the CEO's context window.
"""

# ==============================================================================
# Step 1: Imports and Setup
# ==============================================================================

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Strong model for supervisors, lighter for workers
supervisor_model = ChatOpenAI(model='gpt-4o', temperature=0)
worker_model = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)


# ==============================================================================
# Step 2: Define Mock Tools for the Workers
# ==============================================================================

@tool
def web_search(query: str) -> str:
    """Mock web search tool."""
    return f"Web results for '{query}': Recent industry reports show massive growth and new breakthroughs."

@tool
def wiki_search(query: str) -> str:
    """Mock Wikipedia search tool."""
    return f"Wiki page for '{query}': The foundational history dates back to the 1980s with key academic papers."

@tool
def grammar_check(text: str) -> str:
    """Mock grammar checker tool."""
    return "Grammar check passed. The text is well-structured."


# ==============================================================================
# Step 3: Build the RESEARCH TEAM (Subgraph 1)
# ==============================================================================

class ResearchState(MessagesState):
    next_agent: str

class ResearchDecision(BaseModel):
    next: Literal["web_researcher", "wiki_researcher", "FINISH"]
    reasoning: str

# 1. Workers
web_agent = create_react_agent(worker_model, tools=[web_search])
wiki_agent = create_react_agent(worker_model, tools=[wiki_search])

def web_node(state: ResearchState) -> dict:
    result = web_agent.invoke(state)
    return {"messages": [AIMessage(content=f"[WEB RESEARCHER]: {result['messages'][-1].content}", name="web_researcher")]}

def wiki_node(state: ResearchState) -> dict:
    result = wiki_agent.invoke(state)
    return {"messages": [AIMessage(content=f"[WIKI RESEARCHER]: {result['messages'][-1].content}", name="wiki_researcher")]}

# 2. Supervisor
research_router = supervisor_model.with_structured_output(ResearchDecision)

def research_supervisor(state: ResearchState) -> dict:
    sys_msg = SystemMessage(content=(
        "You are the Research Manager. You manage:\n"
        "- web_researcher: for recent news and web data.\n"
        "- wiki_researcher: for background and history.\n"
        "Ensure both historical and recent data are gathered. Then output FINISH."
    ))
    decision = research_router.invoke([sys_msg] + state["messages"])
    print(f"  🔍 Research Mgr routes to -> {decision.next}")
    return {"next_agent": decision.next}

def route_research(state: ResearchState) -> str:
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]

# 3. Compile Research Subgraph
research_graph = StateGraph(ResearchState)
research_graph.add_node("supervisor", research_supervisor)
research_graph.add_node("web_researcher", web_node)
research_graph.add_node("wiki_researcher", wiki_node)

research_graph.add_edge(START, "supervisor")
research_graph.add_conditional_edges("supervisor", route_research, ["web_researcher", "wiki_researcher", END])
research_graph.add_edge("web_researcher", "supervisor")
research_graph.add_edge("wiki_researcher", "supervisor")

research_team_compiled = research_graph.compile()


# ==============================================================================
# Step 4: Build the WRITING TEAM (Subgraph 2)
# ==============================================================================

class WritingState(MessagesState):
    next_agent: str

class WritingDecision(BaseModel):
    next: Literal["copywriter", "editor", "FINISH"]
    reasoning: str

# 1. Workers
# Copywriter doesn't need external tools, just writes.
copywriter_agent = create_react_agent(worker_model, tools=[])
# Editor uses grammar check
editor_agent = create_react_agent(worker_model, tools=[grammar_check])

def copywriter_node(state: WritingState) -> dict:
    result = copywriter_agent.invoke(state)
    return {"messages": [AIMessage(content=f"[COPYWRITER]: {result['messages'][-1].content}", name="copywriter")]}

def editor_node(state: WritingState) -> dict:
    result = editor_agent.invoke(state)
    return {"messages": [AIMessage(content=f"[EDITOR]: {result['messages'][-1].content}", name="editor")]}

# 2. Supervisor
writing_router = supervisor_model.with_structured_output(WritingDecision)

def writing_supervisor(state: WritingState) -> dict:
    sys_msg = SystemMessage(content=(
        "You are the Writing Manager. You manage:\n"
        "- copywriter: to draft the content based on provided research.\n"
        "- editor: to review the draft and check grammar.\n"
        "Workflow: copywriter drafts -> editor reviews -> FINISH."
    ))
    decision = writing_router.invoke([sys_msg] + state["messages"])
    print(f"  ✍️ Writing Mgr routes to -> {decision.next}")
    return {"next_agent": decision.next}

def route_writing(state: WritingState) -> str:
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]

# 3. Compile Writing Subgraph
writing_graph = StateGraph(WritingState)
writing_graph.add_node("supervisor", writing_supervisor)
writing_graph.add_node("copywriter", copywriter_node)
writing_graph.add_node("editor", editor_node)

writing_graph.add_edge(START, "supervisor")
writing_graph.add_conditional_edges("supervisor", route_writing, ["copywriter", "editor", END])
writing_graph.add_edge("copywriter", "supervisor")
writing_graph.add_edge("editor", "supervisor")

writing_team_compiled = writing_graph.compile()


# ==============================================================================
# Step 5: Build the TOP-LEVEL GRAPH (The Corporate Executive)
# ==============================================================================

class CorporateState(MessagesState):
    next_team: str

class CorporateDecision(BaseModel):
    next: Literal["research_team", "writing_team", "FINISH"]
    reasoning: str

corporate_router = supervisor_model.with_structured_output(CorporateDecision)

def ceo_node(state: CorporateState) -> dict:
    sys_msg = SystemMessage(content=(
        "You are the CEO. You orchestrate two departments to complete the user's request:\n"
        "1. research_team: Gathers raw data.\n"
        "2. writing_team: Writes the final deliverable.\n"
        "Do not answer the prompt yourself. Route to research_team first, then writing_team. "
        "Once writing_team returns the final article, output FINISH."
    ))
    decision = corporate_router.invoke([sys_msg] + state["messages"])
    print(f"\n👔 CEO routes to -> {decision.next.upper()}")
    return {"next_team": decision.next}

def route_corporate(state: CorporateState) -> str:
    if state["next_team"] == "FINISH":
        return END
    return state["next_team"]

# ── Information Hiding Wrappers ──────────────────────────
# When a subgraph finishes, it returns its full internal message history.
# We don't want the CEO to see all the internal team bickering. We just want the final summary.

def run_research_team(state: CorporateState) -> dict:
    # Pass the current state to the subgraph
    result = research_team_compiled.invoke(state)
    # Extract only the last message (the final conclusion of the team)
    final_message = result["messages"][-1]
    return {"messages": [AIMessage(content=f"[RESEARCH DEPT FINAL REPORT]:\n{final_message.content}", name="ResearchDept")]}

def run_writing_team(state: CorporateState) -> dict:
    result = writing_team_compiled.invoke(state)
    final_message = result["messages"][-1]
    return {"messages": [AIMessage(content=f"[WRITING DEPT FINAL DELIVERABLE]:\n{final_message.content}", name="WritingDept")]}


# Compile Corporate Graph
corp_graph = StateGraph(CorporateState)

# Notice we use the wrappers, not the compiled graphs directly
corp_graph.add_node("ceo", ceo_node)
corp_graph.add_node("research_team", run_research_team)
corp_graph.add_node("writing_team", run_writing_team)

corp_graph.add_edge(START, "ceo")
corp_graph.add_conditional_edges("ceo", route_corporate, ["research_team", "writing_team", END])
corp_graph.add_edge("research_team", "ceo")
corp_graph.add_edge("writing_team", "ceo")

corporate_workflow = corp_graph.compile()


# ==============================================================================
# Step 6: Run It
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏢 HIERARCHICAL MAS (NESTED SUPERVISION) — Starting")
    print("=" * 60)

    request = "Write a short blog post about the history and future of AI."
    print(f"User Request: {request}\n")

    # Run the top-level workflow
    # Note: recursion_limit must be higher because of nested graphs
    result = corporate_workflow.invoke({"messages": [HumanMessage(content=request)]}, {"recursion_limit": 50})

    print("\n" + "=" * 60)
    print("🎯 FINAL OUTPUT TO USER")
    print("=" * 60)
    
    # The last message is from the Writing Dept via the CEO loop
    print(result["messages"][-1].content)
    
    print("\n" + "=" * 60)
    print("📁 CEO CONTEXT WINDOW (Information Hiding working)")
    print("=" * 60)
    # Notice that the CEO only sees the high-level reports, not the individual worker chat
    for msg in result["messages"]:
        name = getattr(msg, "name", "User/CEO")
        print(f"[{name.upper()}] Message length: {len(msg.content)} chars")
