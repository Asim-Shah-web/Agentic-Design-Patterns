"""
Autonomous RAG (Self-Corrective & Critique-Driven RAG) - LangGraph Implementation
===================================================================================

This script implements an Autonomous RAG pattern:
1. Autonomous Research Agent: LLM dynamically decides whether to call search tools or draft an answer.
2. Unified Mock Search Tool: Provides a retrieval mechanism over corporate history.
3. Self-Critique Node: Grades document relevance, checks for hallucinations, and evaluates answer completeness.
4. Autonomous Loop Control: Dynamically decides to search more, rewrite queries, or finalize.
"""

import os
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
load_dotenv()

# ==============================================================================
# Dynamic Dual-Provider LLM Setup (OpenAI / Groq)
# ==============================================================================

if os.environ.get("OPENAI_API_KEY"):
    print("[INFO] Detected OpenAI API Key. Running with OpenAI...")
    from langchain_openai import ChatOpenAI
    agent_model = ChatOpenAI(model='gpt-4o', temperature=0)
    critique_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
elif os.environ.get("GROQ_API_KEY"):
    print("[INFO] Detected Groq API Key. Running with Groq...")
    from langchain_groq import ChatGroq
    # Llama 3.3 70B is excellent for autonomous planning and structured output
    agent_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    critique_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
else:
    raise ValueError("Missing credentials. Set OPENAI_API_KEY or GROQ_API_KEY in your .env file.")


# ==============================================================================
# Step 1: Define Mock Database (Multi-Hop Bridge Data)
# ==============================================================================

KNOWLEDGE_BASE = [
    {
        "keywords": ["apex dynamics", "ceo", "executive", "resignation"],
        "source": "Apex Dynamics Press Release (March 2026)",
        "content": (
            "Apex Dynamics Inc. announces that CEO John Doe has submitted his resignation "
            "effective April 30, 2026. The Board of Directors has appointed Sarah Jenkins "
            "as the new CEO, effective May 1, 2026. Jenkins joins from Zenith Corp."
        )
    },
    {
        "keywords": ["sarah jenkins", "zenith corp", "coo"],
        "source": "Corporate Registry - Sarah Jenkins Profile",
        "content": (
            "Sarah Jenkins served as Chief Operating Officer (COO) of Zenith Corp from 2021 to 2026. "
            "Prior to Zenith Corp, she was VP of Engineering at Autonomous Systems Ltd. She holds a "
            "Master's degree in Robotics from MIT."
        )
    },
    {
        "keywords": ["zenith corp", "patents", "intellectual property"],
        "source": "USPTO Database - Zenith Corp Patent Holdings",
        "content": (
            "Zenith Corp holds a total of 52 granted patents in autonomous vehicle guidance, "
            "lidar signal processing, and machine-learning-driven trajectory planning. Key patents "
            "include US-10822194-B2 covering neural network trajectory estimation."
        )
    }
]


# ==============================================================================
# Step 2: Define Retrieval Search Tool
# ==============================================================================

def search_corporate_records(query: str) -> str:
    """Queries corporate records and patent databases for matches."""
    query_lower = query.lower()
    matches = []
    print(f"   [Tool Execution] Searching corporate records for: '{query}'...")
    
    for doc in KNOWLEDGE_BASE:
        # Check if any query terms match keywords or content
        search_space = (" ".join(doc["keywords"]) + " " + doc["content"]).lower()
        if any(term in search_space for term in query_lower.split() if len(term) > 3):
            matches.append(f"Source: [{doc['source']}]\nContent: {doc['content']}")
            
    if not matches:
        return f"No matches found for search query: '{query}'"
    return "\n\n---\n\n".join(matches)


# ==============================================================================
# Step 3: Define Structured Agent and Critique Schemas
# ==============================================================================

# Schema for the Agent's next action
class AgentAction(BaseModel):
    """Represent the next step the autonomous research agent wants to take."""
    action: Literal["search", "draft_response"] = Field(
        description="Select 'search' to run a query against corporate records. Select 'draft_response' if you believe you have gathered all necessary information from search results to completely answer the user query."
    )
    search_query: str = Field(
        default="",
        description="The query string to search for. Must be populated if action is 'search'."
    )
    draft_answer: str = Field(
        default="",
        description="The candidate response to present to the critique panel. Must be populated if action is 'draft_response'."
    )
    reasoning: str = Field(description="Internal reasoning behind choosing this action.")


# Schema for the Critique Node
class CritiqueResult(BaseModel):
    """Result of self-critiquing the candidate answer."""
    status: Literal["approved", "needs_more_data", "needs_regeneration"] = Field(
        description="Choose 'approved' if the answer is grounded and complete. Choose 'needs_more_data' if there are unanswered aspects requiring new searches. Choose 'needs_regeneration' if the answer is hallucinated or logically flawed."
    )
    feedback: str = Field(description="Critique feedback detailing what is missing, hallucinated, or correct.")
    suggested_search: str = Field(
        default="",
        description="Optimized search query to fetch the missing details if status is 'needs_more_data'."
    )

# Bind structured output
structured_agent = agent_model.with_structured_output(AgentAction)
structured_critic = critique_model.with_structured_output(CritiqueResult)


# ==============================================================================
# Step 4: Define Graph State
# ==============================================================================

class AutonomousRAGState(MessagesState):
    """State tracks accumulated facts, tool actions, and critique scores."""
    accumulated_context: List[str]  # Facts collected across all search steps
    agent_action: str               # "search" or "draft_response"
    search_query: str               # Current search string
    candidate_answer: str           # The drafted answer
    critique_feedback: str          # Feedback from the critic
    loop_count: int                 # Loop counter to prevent infinite execution
    final_output: str               # Final verified output to return


# ==============================================================================
# Step 5: Implement Graph Nodes
# ==============================================================================

def autonomous_agent(state: AutonomousRAGState) -> dict:
    """The central reasoning engine. Decides to search or draft based on current knowledge."""
    user_query = state["messages"][-1].content
    context = "\n\n".join(state.get("accumulated_context", []))
    loop_count = state.get("loop_count", 0)
    
    print(f"\n[1. Agent Node] Analyzing state (Loop {loop_count})...")
    
    system_prompt = (
        "You are an Autonomous Research Analyst conducting a multi-hop investigation.\n"
        "Your task is to answer the user query completely and factually.\n\n"
        "RULES:\n"
        "1. Check the 'Accumulated Context' below. If you already have all the facts needed "
        "to answer the original query, select action='draft_response'.\n"
        "2. If you are missing facts, select action='search' and formulate a specific search_query. "
        "Avoid broad or generic queries; search for specific names, companies, or keywords.\n"
        "3. Do not make assumptions. Rely strictly on retrieved facts.\n\n"
        f"USER QUERY: {user_query}\n\n"
        f"ACCUMULATED CONTEXT:\n{context if context else 'None yet.'}"
    )
    
    decision = structured_agent.invoke([SystemMessage(content=system_prompt)])
    print(f"   Decided Action: {decision.action.upper()}")
    print(f"   Reasoning: {decision.reasoning}")
    
    return {
        "agent_action": decision.action,
        "search_query": decision.search_query,
        "candidate_answer": decision.draft_answer,
        "loop_count": loop_count
    }


def execute_search(state: AutonomousRAGState) -> dict:
    """Executes the search query formulated by the agent and appends results to context."""
    query = state["search_query"]
    context_list = state.get("accumulated_context", [])
    
    print(f"\n[2. Search Node] Executing retrieval...")
    search_result = search_corporate_records(query)
    
    # Store source results
    updated_context = context_list + [f"Query: '{query}' -> Results:\n{search_result}"]
    
    return {
        "accumulated_context": updated_context,
        "loop_count": state["loop_count"] + 1
    }


def self_critique(state: AutonomousRAGState) -> dict:
    """Critiques the candidate answer for grounding (hallucination) and completeness."""
    user_query = state["messages"][-1].content
    context = "\n\n".join(state.get("accumulated_context", []))
    candidate = state["candidate_answer"]
    
    print("\n[3. Critique Node] Evaluating candidate response...")
    
    prompt = (
        "You are an Independent Quality Assurance Critic. Evaluate the candidate response.\n\n"
        "CRITERIA:\n"
        "1. GROUNDING: Is every statement in the candidate response directly supported by the Accumulated Context? "
        "If there are hallucinated numbers, dates, names, or assumptions, mark status='needs_regeneration'.\n"
        "2. COMPLETENESS: Does the candidate response fully answer all parts of the User Query? "
        "If some parts of the question are unanswered due to missing context, mark status='needs_more_data' "
        "and suggest a specific query in 'suggested_search'.\n\n"
        f"User Query: {user_query}\n\n"
        f"Accumulated Context:\n{context}\n\n"
        f"Candidate Response:\n{candidate}"
    )
    
    critique = structured_critic.invoke([HumanMessage(content=prompt)])
    
    print(f"   Critique Verdict: {critique.status.upper()}")
    print(f"   Critique Feedback: {critique.feedback}")
    if critique.suggested_search:
        print(f"   Suggested Next Search: '{critique.suggested_search}'")
        
    # Update search query for the next loop if needed
    update = {
        "critique_feedback": critique.feedback,
        "agent_action": "search" if critique.status == "needs_more_data" else ("draft_response" if critique.status == "needs_regeneration" else "finalize"),
        "final_output": candidate if critique.status == "approved" else ""
    }
    
    if critique.status == "needs_more_data":
        update["search_query"] = critique.suggested_search
        
    return update


def finalize_answer(state: AutonomousRAGState) -> dict:
    """Concludes the graph, producing the finalized research output."""
    print("\n[4. Finalize Node] Preparing verified response.")
    ans = state["final_output"]
    return {
        "messages": [AIMessage(content=ans, name="ResearchAnalyst")]
    }


# ==============================================================================
# Step 6: Define Conditional Routers (State Machine Logic)
# ==============================================================================

def route_agent_decision(state: AutonomousRAGState) -> str:
    """Routes based on the agent's action decision."""
    if state["loop_count"] >= 6:
        print("\n[Loop Guard Alert] Maximum loops reached. Forcing final answer.")
        return "finalize"
        
    if state["agent_action"] == "search":
        return "execute_search"
    return "self_critique"

def route_critique_decision(state: AutonomousRAGState) -> str:
    """Routes based on the critic's grading result."""
    if state["agent_action"] == "finalize":
        return "finalize"
    elif state["agent_action"] == "search":
        # Loop back directly to execute search with the critic's suggested query
        return "execute_search"
    else:
        # Loop back to agent node for regeneration
        return "autonomous_agent"


# Build the graph
workflow = StateGraph(AutonomousRAGState)

# Add nodes
workflow.add_node("autonomous_agent", autonomous_agent)
workflow.add_node("execute_search", execute_search)
workflow.add_node("self_critique", self_critique)
workflow.add_node("finalize_answer", finalize_answer)

# Wire edges
workflow.add_edge(START, "autonomous_agent")

workflow.add_conditional_edges(
    "autonomous_agent",
    route_agent_decision,
    {
        "execute_search": "execute_search",
        "self_critique": "self_critique",
        "finalize": "finalize_answer"
    }
)

workflow.add_edge("execute_search", "autonomous_agent")

workflow.add_conditional_edges(
    "self_critique",
    route_critique_decision,
    {
        "finalize": "finalize_answer",
        "execute_search": "execute_search",
        "autonomous_agent": "autonomous_agent"
    }
)

workflow.add_edge("finalize_answer", END)

# Compile
compiled_graph = workflow.compile()


# ==============================================================================
# Step 7: Execution and Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AUTONOMOUS RAG (INTELLIGENCE DOSSIER ANALYST) - Start")
    print("=" * 80)
    
    # A complex multi-hop query requiring discovery of bridge entities and patent data
    user_query = (
        "Find out who the current CEO of Apex Dynamics is, "
        "where they worked previously, and how many patents that previous company holds."
    )
    
    print(f"Investigation Query:\n\"{user_query}\"\n")
    
    inputs = {
        "messages": [HumanMessage(content=user_query)],
        "accumulated_context": [],
        "agent_action": "search",
        "search_query": "",
        "candidate_answer": "",
        "critique_feedback": "",
        "loop_count": 0,
        "final_output": ""
    }
    
    # Run the graph
    result = compiled_graph.invoke(inputs, {"recursion_limit": 30})
    
    print("\n" + "=" * 80)
    print("FINAL VERIFIED dossier")
    print("=" * 80)
    print(result["messages"][-1].content)
    print("=" * 80)
    
    # Audit trail printout
    print("\nCONTEXT COLLECTION AUDIT TRAIL")
    print("=" * 80)
    for idx, ctx in enumerate(result.get("accumulated_context", []), 1):
        print(f"\n[Retrieval #{idx}]\n{ctx}")
    print("=" * 80 + "\n")
