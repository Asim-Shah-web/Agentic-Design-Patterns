"""
Self-RAG (Self-Reflective Retrieval-Augmented Generation) - LangGraph Implementation
=====================================================================================

This script implements a Self-RAG pattern:
1. Retrieval Router: Decides if the query needs retrieval.
2. Context Retriever: Pulls data from mock financial files.
3. Relevance Critic (IsRel): Filters out irrelevant documents.
4. Candidate Generator: Drafts responses.
5. Grounding Critic (IsSup): Checks for hallucinations in draft.
6. Utility Critic (IsUse): Checks if the answer completely addresses the query.
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
    eval_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    gen_model = ChatOpenAI(model='gpt-4o', temperature=0.1)
elif os.environ.get("GROQ_API_KEY"):
    print("[INFO] Detected Groq API Key. Running with Groq...")
    from langchain_groq import ChatGroq
    # Llama 3.3 70B is excellent for structured critique and self-reflection
    eval_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    gen_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1)
else:
    raise ValueError("Missing credentials. Set OPENAI_API_KEY or GROQ_API_KEY in your .env file.")


# ==============================================================================
# Step 1: Define Mock Database (Financial Reports)
# ==============================================================================

FINANCIAL_REPORT_DB = [
    {
        "topic": "Revenue and Operations",
        "content": (
            "Nexus Corp Q3 2025 Balance Sheet details:\n"
            "- Q3 Revenue: $120.4 Million USD (up 14% year-over-year).\n"
            "- Operating Expenses: Increased by $3.2 Million USD due to global marketing campaign costs.\n"
            "- Operating Profit (EBITDA): $28.5 Million USD."
        )
    },
    {
        "topic": "Principal Risks",
        "content": (
            "Nexus Corp Q3 2025 Risk Disclosure:\n"
            "- Principal Risks: High dependency on overseas shipping container availability, "
            "which may cause inventory delays of up to 45 days in Q4.\n"
            "- Secondary Risks: Regulatory changes in European environmental packaging guidelines."
        )
    }
]


# ==============================================================================
# Step 2: Define Retrieval Helpers
# ==============================================================================

def retrieve_financial_docs(query: str) -> List[str]:
    """Retrieves document snippets matching keywords in query."""
    query_lower = query.lower()
    results = []
    for doc in FINANCIAL_REPORT_DB:
        searchable = f"{doc['topic']} {doc['content']}".lower()
        if any(word in searchable for word in query_lower.split() if len(word) > 3):
            results.append(f"[{doc['topic']}]: {doc['content']}")
    return results


# ==============================================================================
# Step 3: Define Structured Self-Reflection Schemas
# ==============================================================================

class RetrieveDecision(BaseModel):
    """Evaluation result on whether the query needs external retrieval."""
    needs_retrieval: bool = Field(
        description="True if answering the query requires specific external knowledge, corporate documents, or figures. False if it is a general greeting, chit-chat, or common knowledge."
    )
    reasoning: str = Field(description="Explanation of the retrieval necessity.")

class RelevanceScore(BaseModel):
    """Evaluation result for document relevance (IsRel)."""
    is_relevant: bool = Field(
        description="True if the document contains info directly relevant to answering the query, False otherwise."
    )
    reasoning: str = Field(description="Reasoning behind relevance decision.")

class GroundednessScore(BaseModel):
    """Evaluation result checking if the response is supported by the context (IsSup)."""
    is_supported: bool = Field(
        description="True if every fact, statistic, and statement in the candidate answer is explicitly supported by the retrieved context. False if there are hallucinations or unsupported claims."
    )
    reasoning: str = Field(description="Verification details checking statements against context.")

class UtilityScore(BaseModel):
    """Evaluation result checking if the response fully answers the query (IsUse)."""
    is_useful: bool = Field(
        description="True if the candidate answer completely and directly addresses the user query. False if the answer is incomplete, too brief, or misses key requested dimensions."
    )
    reasoning: str = Field(description="Evaluation of completeness.")


# Bind structured output schemas to LLM
structured_router = eval_model.with_structured_output(RetrieveDecision)
structured_relevance_grader = eval_model.with_structured_output(RelevanceScore)
structured_groundedness_grader = eval_model.with_structured_output(GroundednessScore)
structured_utility_grader = eval_model.with_structured_output(UtilityScore)


# ==============================================================================
# Step 4: Define Self-RAG State
# ==============================================================================

class SelfRAGState(MessagesState):
    """State containing retrieved documents, critiques, candidate drafts, and loop control."""
    needs_retrieval: bool      # Route flag: True (retrieve) / False (direct)
    retrieved_docs: List[str]  # Chunks passing IsRel check
    candidate_answer: str      # Candidate generation
    is_supported: bool         # Grounding rating (IsSup)
    is_useful: bool            # Utility rating (IsUse)
    loop_count: int            # Prevents infinite evaluation loops
    generation: str            # Final response output


# ==============================================================================
# Step 5: Implement Graph Nodes
# ==============================================================================

def decide_retrieval_node(state: SelfRAGState) -> dict:
    """Decides if query needs retrieval check (First check)."""
    user_msg = state["messages"][-1].content
    print(f"\n[1. Router] Evaluating retrieval necessity for: '{user_msg}'")
    
    prompt = [
        SystemMessage(content="Determine if the user query requires fetching private corporate financial documents."),
        HumanMessage(content=user_msg)
    ]
    decision = structured_router.invoke(prompt)
    print(f"   Needs Retrieval: {decision.needs_retrieval} (Reason: {decision.reasoning})")
    
    return {
        "needs_retrieval": decision.needs_retrieval,
        "retrieved_docs": [],
        "loop_count": 0
    }


def direct_response_node(state: SelfRAGState) -> dict:
    """Answers simple general knowledge or greetings directly."""
    user_msg = state["messages"][-1].content
    print("\n[Node: Direct Response] Replying without database context...")
    
    prompt = f"Answer the following query directly and politely: {user_msg}"
    response = gen_model.invoke([HumanMessage(content=prompt)])
    
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content, name="FinancialSupport")]
    }


def retrieve_node(state: SelfRAGState) -> dict:
    """Retrieves document chunks matching user query."""
    user_query = state["messages"][-1].content
    print(f"\n[2. Retriever] Searching financial database for: '{user_query}'")
    
    docs = retrieve_financial_docs(user_query)
    print(f"   Found {len(docs)} document candidate(s).")
    
    return {"retrieved_docs": docs}


def evaluate_relevance_node(state: SelfRAGState) -> dict:
    """Critiques document relevance (IsRel token evaluation)."""
    user_query = state["messages"][-1].content
    docs = state.get("retrieved_docs", [])
    
    print("\n[3. IsRel Grader] Critiquing context relevance...")
    
    relevant_docs = []
    for doc in docs:
        prompt = (
            "Decide if this document segment is relevant to answering the query.\n\n"
            f"User Query: {user_query}\n"
            f"Document:\n{doc}"
        )
        score = structured_relevance_grader.invoke([HumanMessage(content=prompt)])
        
        if score.is_relevant:
            print(f"   [IsRel: PASS] {doc[:60]}...")
            relevant_docs.append(doc)
        else:
            print(f"   [IsRel: FAIL] {doc[:60]}...")
            
    return {"retrieved_docs": relevant_docs}


def generate_candidate_node(state: SelfRAGState) -> dict:
    """Drafts candidate response using filtered context."""
    user_query = state["messages"][-1].content
    docs = state.get("retrieved_docs", [])
    loop_count = state.get("loop_count", 0) + 1
    
    print(f"\n[4. Generator] Drafting candidate response (Attempt {loop_count})...")
    
    context = "\n\n".join(docs) if docs else "No retrieved context available."
    prompt = (
        "You are an expert investment auditor. Draft a response to the user query based "
        "only on the provided context. Be precise and cite metrics carefully. Do not assume or extrapolate.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {user_query}"
    )
    
    response = gen_model.invoke([HumanMessage(content=prompt)])
    
    return {
        "candidate_answer": response.content,
        "loop_count": loop_count
    }


def evaluate_groundedness_node(state: SelfRAGState) -> dict:
    """Critiques candidate response support (IsSup token evaluation)."""
    candidate = state["candidate_answer"]
    context = "\n\n".join(state.get("retrieved_docs", []))
    
    print("\n[5. IsSup Grader] Critiquing draft groundedness...")
    
    prompt = (
        "Verify if all claims in the candidate response are supported by facts in the context. "
        "If there are figures or names in the response that do not match the context, mark is_supported=False.\n\n"
        f"Context:\n{context}\n\n"
        f"Candidate Response:\n{candidate}"
    )
    
    score = structured_groundedness_grader.invoke([HumanMessage(content=prompt)])
    print(f"   [IsSup Verdict] Grounded: {score.is_supported} (Reason: {score.reasoning})")
    
    return {"is_supported": score.is_supported}


def evaluate_utility_node(state: SelfRAGState) -> dict:
    """Critiques response usefulness and completeness (IsUse token evaluation)."""
    user_query = state["messages"][-1].content
    candidate = state["candidate_answer"]
    
    print("\n[6. IsUse Grader] Critiquing answer utility...")
    
    prompt = (
        "Assess whether the candidate response completely answers all aspects of the user query.\n\n"
        f"User Query: {user_query}\n\n"
        f"Candidate Response:\n{candidate}"
    )
    
    score = structured_utility_grader.invoke([HumanMessage(content=prompt)])
    print(f"   [IsUse Verdict] Complete: {score.is_useful} (Reason: {score.reasoning})")
    
    return {"is_useful": score.is_useful}


def deliver_response_node(state: SelfRAGState) -> dict:
    """Delivers final critique-verified response."""
    print("\n[7. Final Deliver] Response verified. Releasing dossier.")
    ans = state["candidate_answer"]
    return {
        "generation": ans,
        "messages": [AIMessage(content=ans, name="AuditorResponse")]
    }


# ==============================================================================
# Step 6: Define Conditional Routers (State Machine Logic)
# ==============================================================================

def route_initial_decision(state: SelfRAGState) -> str:
    """Routes query based on initial retrieval router decision."""
    if state["needs_retrieval"]:
        return "retrieve"
    return "direct_response"

def route_relevance_decision(state: SelfRAGState) -> str:
    """Ensures we have relevant documents before generating."""
    if not state.get("retrieved_docs"):
        # Loop back to retrieve (or terminate if loop limit hit)
        if state.get("loop_count", 0) >= 3:
            return "direct_response"
        return "retrieve"
    return "generate_candidate"

def route_groundedness_decision(state: SelfRAGState) -> str:
    """Branches based on hallucination grade (IsSup)."""
    if state["is_supported"]:
        return "evaluate_utility"
        
    # Grounding fails -> loop back to generate candidate (regenerate)
    if state.get("loop_count", 0) >= 3:
        return "evaluate_utility"  # Force bypass if repeating
    return "generate_candidate"

def route_utility_decision(state: SelfRAGState) -> str:
    """Branches based on answer completeness grade (IsUse)."""
    if state["is_useful"] or state.get("loop_count", 0) >= 3:
        return "deliver_response"
        
    # Utility fails -> Loop back to retrieval (try fetching different sections)
    return "retrieve"


# Assemble the graph
workflow = StateGraph(SelfRAGState)

# Add Nodes
workflow.add_node("decide_retrieval", decide_retrieval_node)
workflow.add_node("direct_response", direct_response_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate_relevance", evaluate_relevance_node)
workflow.add_node("generate_candidate", generate_candidate_node)
workflow.add_node("evaluate_groundedness", evaluate_groundedness_node)
workflow.add_node("evaluate_utility", evaluate_utility_node)
workflow.add_node("deliver_response", deliver_response_node)

# Wire Edges
workflow.add_edge(START, "decide_retrieval")

workflow.add_conditional_edges(
    "decide_retrieval",
    route_initial_decision,
    {
        "retrieve": "retrieve",
        "direct_response": "direct_response"
    }
)

workflow.add_edge("retrieve", "evaluate_relevance")

workflow.add_conditional_edges(
    "evaluate_relevance",
    route_relevance_decision,
    {
        "retrieve": "retrieve",
        "generate_candidate": "generate_candidate"
    }
)

workflow.add_edge("generate_candidate", "evaluate_groundedness")

workflow.add_conditional_edges(
    "evaluate_groundedness",
    route_groundedness_decision,
    {
        "evaluate_utility": "evaluate_utility",
        "generate_candidate": "generate_candidate"
    }
)

workflow.add_conditional_edges(
    "evaluate_utility",
    route_utility_decision,
    {
        "deliver_response": "deliver_response",
        "retrieve": "retrieve"
    }
)

workflow.add_edge("deliver_response", END)
workflow.add_edge("direct_response", END)

# Compile
compiled_self_rag = workflow.compile()


# ==============================================================================
# Step 7: Execution and Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SELF-RAG (AUDITOR PROCESSOR) - Start")
    print("=" * 80)
    
    # Test Queries:
    # 1. Chit-chat (Direct response)
    # 2. Comprehensive auditor report (Triggers retrieval and self-reflective checks)
    
    test_queries = [
        "Good morning support, I hope you are well. Can you show me standard contact details?",
        "What was Nexus Corp's Q3 2025 revenue growth rate and what are their primary Q4 risks?"
    ]
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n\n[TEST CASE #{idx}] Query: \"{query}\"")
        print("-" * 80)
        
        inputs = {
            "messages": [HumanMessage(content=query)],
            "needs_retrieval": False,
            "retrieved_docs": [],
            "candidate_answer": "",
            "is_supported": False,
            "is_useful": False,
            "loop_count": 0,
            "generation": ""
        }
        
        result = compiled_self_rag.invoke(inputs, {"recursion_limit": 30})
        
        print("\n[RESULT]")
        print(f"Final Output:\n{result['generation']}")
        print("=" * 80)
