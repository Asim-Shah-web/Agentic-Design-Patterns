"""
Adaptive RAG (Adaptive Retrieval-Augmented Generation) - LangGraph Implementation
===================================================================================

This script implements an Adaptive RAG pattern:
1. Query Classifier: Classifies user queries into "direct_response", "local_rag", or "web_search".
2. Local RAG Retrieval: Queries a local knowledge base of specifications and internal docs.
3. Web Search Retrieval: Simulates a live external status / troubleshooting search.
4. Document Grader: Filters out irrelevant retrieved context.
5. Hallucination Grader: Checks if the generated answer is faithful to the context.
6. Answer Grader: Assesses whether the response actually answers the query.
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
    router_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    grader_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    generation_model = ChatOpenAI(model='gpt-4o', temperature=0.1)
elif os.environ.get("GROQ_API_KEY"):
    print("[INFO] Detected Groq API Key. Running with Groq...")
    from langchain_groq import ChatGroq
    # Llama 3.3 70B is excellent for structured output and routing
    router_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    grader_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    generation_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1)
else:
    raise ValueError("Missing credentials. Set OPENAI_API_KEY or GROQ_API_KEY in your .env file.")


# ==============================================================================
# Step 1: Define Mock Knowledge Bases (Local & Web mock)
# ==============================================================================

# Local VM specifications database
LOCAL_DOCS_DB = [
    {
        "title": "VM Tier-1 Specifications",
        "content": "VM Tier-1 (Basic Compute) offers 1 vCPU, 2GB RAM, and 20GB SSD storage. It is designed for lightweight testing and development. Bandwidth limit: 1TB/month. Cost: $5/month."
    },
    {
        "title": "VM Tier-2 Specifications",
        "content": "VM Tier-2 (Standard Compute) offers 2 vCPUs, 8GB RAM, and 80GB SSD storage. It is designed for production web applications and lightweight databases. Bandwidth limit: 5TB/month. Cost: $20/month."
    },
    {
        "title": "VM Tier-3 Specifications",
        "content": "VM Tier-3 (High Performance Compute) offers 8 vCPUs, 32GB RAM, and 300GB NVMe SSD storage. Optimized for data-intensive workloads and high-traffic databases. Bandwidth limit: 10TB/month. Cost: $80/month."
    },
    {
        "title": "CloudFlow Support Contacts",
        "content": "For billing inquiries, email billing@cloudflow.io. For general support, open a ticket at help.cloudflow.io. Customer support hotline: +1-800-555-0199."
    }
]

# Web/Status mock database (for troubleshooting and real-time events)
WEB_SEARCH_DB = [
    {
        "source": "CloudFlow Live Status Page (June 2, 2026)",
        "content": "System Alert: US-East-1 region network congestion detected. Starting at 14:30 UTC, packet loss up to 15% is affecting VM instances. Engineers are rerouting traffic. Estimated resolution: 18:00 UTC."
    },
    {
        "source": "AWS Tech Status Forum (June 2026)",
        "content": "Reported issue: Recent kernel update v6.1.8-generic causes boot failure on Ubuntu 22.04 LTS instances configured with volume-backed root storage. Workaround: Rollback to kernel v6.1.7."
    }
]


# ==============================================================================
# Step 2: Define Retrieval Helper Functions
# ==============================================================================

def retrieve_local_docs(query: str) -> List[str]:
    """Retrieves relevant internal docs from the local DB using simple keyword matching."""
    query_lower = query.lower()
    matches = []
    for doc in LOCAL_DOCS_DB:
        searchable = f"{doc['title']} {doc['content']}".lower()
        if any(word in searchable for word in query_lower.split() if len(word) > 3):
            matches.append(f"[{doc['title']}]: {doc['content']}")
    return matches

def retrieve_web_search(query: str) -> List[str]:
    """Retrieves relevant real-time status/troubleshooting from mock web DB."""
    query_lower = query.lower()
    matches = []
    for doc in WEB_SEARCH_DB:
        searchable = f"{doc['source']} {doc['content']}".lower()
        if any(word in searchable for word in query_lower.split() if len(word) > 3):
            matches.append(f"[{doc['source']}]: {doc['content']}")
    return matches


# ==============================================================================
# Step 3: Define Structured Routing & Grading Schemas
# ==============================================================================

class QueryRoute(BaseModel):
    """Structured output representing the classified query route."""
    route: Literal["direct_response", "local_rag", "web_search"] = Field(
        description="Choose 'direct_response' for greetings/chit-chat. Choose 'local_rag' for specific internal documents, VM specs, and contacts. Choose 'web_search' for real-time outages, tech status, or complex troubleshooting queries."
    )
    reasoning: str = Field(description="Brief explanation of the routing classification.")

class DocumentRelevance(BaseModel):
    """Structured output for grading a document's relevance to the query."""
    relevance: Literal["yes", "no"] = Field(
        description="Score 'yes' if the document contains information relevant to the user query, otherwise score 'no'."
    )
    reasoning: str = Field(description="Explanation of the relevance score.")

class HallucinationScore(BaseModel):
    """Structured output for checking factual alignment of generation with retrieved context."""
    is_grounded: bool = Field(
        description="True if the generated answer is completely grounded in and supported by the retrieved facts, False otherwise."
    )
    reasoning: str = Field(description="Explanation of alignment/gaps between answer and context.")

class AnswerCompleteness(BaseModel):
    """Structured output for grading if the answer addresses all parts of the user query."""
    is_complete: bool = Field(
        description="True if the generated answer completely and directly addresses the user query, False otherwise."
    )
    reasoning: str = Field(description="Explanation of what is missing or if it is fully answered.")


# Bind structured outputs to models
structured_router = router_model.with_structured_output(QueryRoute)
structured_doc_grader = grader_model.with_structured_output(DocumentRelevance)
structured_hallucination_grader = grader_model.with_structured_output(HallucinationScore)
structured_answer_grader = grader_model.with_structured_output(AnswerCompleteness)


# ==============================================================================
# Step 4: Define the Adaptive RAG State
# ==============================================================================

class AdaptiveRAGState(MessagesState):
    """State containing routing details, retrieved content, evaluations, and final response."""
    route: str                 # "direct_response", "local_rag", "web_search"
    retrieved_docs: List[str]  # Loaded document snippets
    failed_attempts: int       # Tracker for local RAG failure/rewrites
    rewritten_query: str       # Store modified query if local RAG fails
    generation: str            # Final compiled LLM response


# ==============================================================================
# Step 5: Implement Graph Nodes
# ==============================================================================

def analyze_query(state: AdaptiveRAGState) -> dict:
    """Classifies user input to determine the optimal execution path."""
    user_msg = state["messages"][-1].content
    print(f"\n[1. Router] Analyzing query: '{user_msg}'")
    
    prompt = [
        SystemMessage(content="You are a routing agent for a cloud support desk. Classify the user query into the correct path."),
        HumanMessage(content=user_msg)
    ]
    
    decision = structured_router.invoke(prompt)
    print(f"   Route Decided: {decision.route.upper()} (Reason: {decision.reasoning})")
    
    return {
        "route": decision.route,
        "failed_attempts": state.get("failed_attempts", 0)
    }

def direct_response(state: AdaptiveRAGState) -> dict:
    """Answers simple conversational/chit-chat queries directly."""
    user_msg = state["messages"][-1].content
    print("\n[Node: Direct Response] Processing direct reply...")
    
    prompt = (
        "You are an assistant for CloudFlow support. Answer the user query politely. "
        "Do not invent technical documentation specs if not present in your general knowledge.\n\n"
        f"Query: {user_msg}"
    )
    response = generation_model.invoke([HumanMessage(content=prompt)])
    
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content, name="SupportAssistant")]
    }

def retrieve_local(state: AdaptiveRAGState) -> dict:
    """Retrieves context from internal Vector database."""
    query = state.get("rewritten_query") or state["messages"][-1].content
    print(f"\n[Node: Local Retrieval] Searching local docs for: '{query}'")
    
    docs = retrieve_local_docs(query)
    print(f"   Found {len(docs)} local documents.")
    
    return {"retrieved_docs": docs}

def retrieve_web(state: AdaptiveRAGState) -> dict:
    """Retrieves status/troubleshooting info from live web sources."""
    query = state.get("rewritten_query") or state["messages"][-1].content
    print(f"\n[Node: Web Search] Searching status databases for: '{query}'")
    
    docs = retrieve_web_search(query)
    print(f"   Found {len(docs)} matching online postings.")
    
    return {"retrieved_docs": docs}

def grade_documents(state: AdaptiveRAGState) -> dict:
    """Grades each retrieved document for relevance to the user's query."""
    user_query = state["messages"][-1].content
    docs = state.get("retrieved_docs", [])
    
    print("\n[Node: Document Grader] Grading retrieved context...")
    
    relevant_docs = []
    for doc in docs:
        prompt = (
            "Determine if the following document is relevant to answering the user query.\n\n"
            f"User Query: {user_query}\n"
            f"Document:\n{doc}"
        )
        grade = structured_doc_grader.invoke([HumanMessage(content=prompt)])
        if grade.relevance == "yes":
            print(f"   [Relevance: YES] {doc[:60]}...")
            relevant_docs.append(doc)
        else:
            print(f"   [Relevance: NO] {doc[:60]}...")
            
    return {
        "retrieved_docs": relevant_docs,
        # If no documents are relevant, mark it to trigger routing shift
        "route": "local_rag" if relevant_docs else "rewrite_query"
    }

def rewrite_query(state: AdaptiveRAGState) -> dict:
    """Rewrites query to improve retrieval precision before shifting to web search."""
    user_query = state["messages"][-1].content
    attempts = state.get("failed_attempts", 0) + 1
    
    print(f"\n[Node: Query Rewriter] Local retrieval failed. Rewriting query for broader search.")
    
    prompt = (
        "You are a search query optimizer. The user's query could not find matching documents "
        "in our internal database. Rewrite the query to optimize it for external search and troubleshooting "
        "status databases. Keep it concise.\n\n"
        f"Original query: {user_query}"
    )
    rewritten = router_model.invoke([HumanMessage(content=prompt)])
    print(f"   Rewritten Query: '{rewritten.content}'")
    
    return {
        "rewritten_query": rewritten.content,
        "failed_attempts": attempts,
        "route": "web_search"  # Pivot to web search
    }

def generate_answer(state: AdaptiveRAGState) -> dict:
    """Generates the final response based on validated documents."""
    user_query = state["messages"][-1].content
    docs = state.get("retrieved_docs", [])
    
    print("\n[Node: Generator] Constructing grounded response...")
    
    context = "\n\n".join(docs) if docs else "No retrieved context available."
    prompt = (
        "You are CloudFlow Technical Support. Answer the user query using only the "
        "provided context. If the answer cannot be determined from the context, "
        "state that clearly.\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"User Query: {user_query}"
    )
    
    response = generation_model.invoke([HumanMessage(content=prompt)])
    
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content, name="TechnicalSupport")]
    }


# ==============================================================================
# Step 6: Define Conditional Routers (State Machine Logic)
# ==============================================================================

def route_initial_decision(state: AdaptiveRAGState) -> str:
    """Routes initial query based on classifier analysis."""
    return state["route"]

def route_document_grade(state: AdaptiveRAGState) -> str:
    """Evaluates relevance grading: if no relevant documents, rewrite query."""
    if state["route"] == "rewrite_query":
        return "rewrite_query"
    return "generate_answer"

def grade_generation_and_conclude(state: AdaptiveRAGState) -> str:
    """
    Self-Correction Loop: Checks for hallucinations (groundedness) and
    completeness (answering the question).
    """
    user_query = state["messages"][-1].content
    context = "\n\n".join(state.get("retrieved_docs", []))
    generation = state["generation"]
    
    print("\n[Step: Self-Correction Grader] Analyzing generation...")
    
    # 1. Hallucination Check
    hallucination_prompt = (
        "Evaluate whether the generated answer is completely grounded in the retrieved context.\n"
        "It must not contain facts or assumptions not present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Generated Answer:\n{generation}"
    )
    hallucination = structured_hallucination_grader.invoke([HumanMessage(content=hallucination_prompt)])
    if not hallucination.is_grounded:
        print(f"   [Grader Alert] Hallucination detected! (Reason: {hallucination.reasoning})")
        return "generate_answer"  # Loop back to regenerate
    
    print("   [Grader: Pass] Answer is grounded (no hallucinations).")
    
    # 2. Answer Completeness Check
    completeness_prompt = (
        "Evaluate whether the generated answer fully and directly answers the user's question.\n\n"
        f"User Query: {user_query}\n\n"
        f"Generated Answer:\n{generation}"
    )
    completeness = structured_answer_grader.invoke([HumanMessage(content=completeness_prompt)])
    if not completeness.is_complete:
        print(f"   [Grader Alert] Answer is incomplete! (Reason: {completeness.reasoning})")
        # Try a query rewrite or escalation if we haven't failed already
        if state.get("failed_attempts", 0) < 1:
            return "rewrite_query"
        else:
            print("   [Grader Notice] Max escalation attempts reached. Finalizing fallback reply.")
            return "end"
            
    print("   [Grader: Pass] Answer fully addresses the user query.")
    return "end"


# ==============================================================================
# Step 7: Assemble the Graph
# ==============================================================================

workflow = StateGraph(AdaptiveRAGState)

# Add Nodes
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("direct_response", direct_response)
workflow.add_node("retrieve_local", retrieve_local)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_web", retrieve_web)
workflow.add_node("generate_answer", generate_answer)

# Wire Edges
workflow.add_edge(START, "analyze_query")

workflow.add_conditional_edges(
    "analyze_query",
    route_initial_decision,
    {
        "direct_response": "direct_response",
        "local_rag": "retrieve_local",
        "web_search": "retrieve_web"
    }
)

workflow.add_edge("retrieve_local", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_document_grade,
    {
        "rewrite_query": "rewrite_query",
        "generate_answer": "generate_answer"
    }
)

workflow.add_edge("rewrite_query", "retrieve_web")
workflow.add_edge("retrieve_web", "generate_answer")

workflow.add_conditional_edges(
    "generate_answer",
    grade_generation_and_conclude,
    {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query",
        "end": END
    }
)

workflow.add_edge("direct_response", END)

# Compile
compiled_graph = workflow.compile()


# ==============================================================================
# Step 8: Execution and Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ADAPTIVE RAG (SUPPORT CO-PILOT) - Starting Execution Demo")
    print("=" * 80)
    
    # Test Queries representing different paths:
    # 1. Direct Response (Greetings / Chit-chat)
    # 2. Local RAG (Specific VM query - found in local docs)
    # 3. Escalated/Web RAG (Troubleshooting outage - not in local docs, routes to web status)
    
    test_queries = [
        "Hello! I hope you are having a nice day. Who do I email if I have a question about my bill?",
        "What SSD capacity and memory is provided under VM Tier-2?",
        "My instance is experiencing heavy packet loss in us-east-1 right now. Is there a status alert?"
    ]
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n\n[TEST CASE #{idx}] Query: \"{query}\"")
        print("-" * 80)
        
        inputs = {
            "messages": [HumanMessage(content=query)],
            "route": "direct_response",
            "retrieved_docs": [],
            "failed_attempts": 0,
            "rewritten_query": "",
            "generation": ""
        }
        
        result = compiled_graph.invoke(inputs, {"recursion_limit": 25})
        
        print("\n[RESULT]")
        print(f"Final Output:\n{result['generation']}")
        print("=" * 80)
