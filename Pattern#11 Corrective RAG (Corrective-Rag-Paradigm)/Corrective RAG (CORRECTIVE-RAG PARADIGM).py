"""
Corrective RAG (Corrective Retrieval-Augmented Generation) - LangGraph Implementation
=====================================================================================

This script implements a Corrective RAG (CRAG) pattern:
1. Local Retriever: Searches a local database.
2. Document Evaluator / Grader: Evaluates document relevance to check if local data is incorrect/stale.
3. Query Transformer: Rewrites queries when local documents fail grading.
4. Web Search Fallback: Searches external databases to retrieve correct/current data.
5. Generator: Compiles the finalized corrected response.
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
    transform_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    gen_model = ChatOpenAI(model='gpt-4o', temperature=0.1)
elif os.environ.get("GROQ_API_KEY"):
    print("[INFO] Detected Groq API Key. Running with Groq...")
    from langchain_groq import ChatGroq
    # Llama 3.3 70B is excellent for grading and routing logic
    eval_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    transform_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    gen_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1)
else:
    raise ValueError("Missing credentials. Set OPENAI_API_KEY or GROQ_API_KEY in your .env file.")


# ==============================================================================
# Step 1: Define Mock Database (Local Vector DB & Web status DB)
# ==============================================================================

# Local product manual index
LOCAL_FAQ_DB = [
    {
        "product": "SmartPlug v1",
        "doc": "ElectroNova SmartPlug v1 Product Details: Supports loads up to 1000W. Price: $14.99. Standard on/off timing schedule via ElectroNova Basic App. Connectivity: 2.4GHz Wi-Fi only."
    },
    {
        "product": "SmartPlug v2",
        "doc": "ElectroNova SmartPlug v2 Product Details: Supports loads up to 1500W. Price: $19.99. Advanced scheduling, countdown timers, and voice assistant integration (Alexa/Google Home). Connectivity: Dual-band Wi-Fi."
    }
]

# External Web Search index (recent news/launches)
WEB_SEARCH_DB = [
    {
        "query_keywords": ["smartplug pro", "smartplug", "features", "specs"],
        "source": "TechCrunch Product Launch Article (Yesterday)",
        "content": (
            "ElectroNova launches 'SmartPlug Pro' for heavy-duty appliances. "
            "Features energy-monitoring dashboards, overload protection up to 1800W, "
            "and offline local network control. Priced at $29.99, it is available immediately."
        )
    }
]


# ==============================================================================
# Step 2: Define Retrieval Helper Functions
# ==============================================================================

def retrieve_local_faq(query: str) -> List[str]:
    """Retrieves documents matching keywords in the local DB."""
    query_lower = query.lower()
    results = []
    for doc in LOCAL_FAQ_DB:
        if any(word in doc["doc"].lower() for word in query_lower.split() if len(word) > 3):
            results.append(doc["doc"])
    return results

def execute_external_web_search(query: str) -> List[str]:
    """Simulates querying external search index for out-of-distribution queries."""
    query_lower = query.lower()
    results = []
    print(f"   [Web Search] Executing live search for query: '{query}'...")
    for doc in WEB_SEARCH_DB:
        combined_keywords = " ".join(doc["query_keywords"]) + " " + doc["content"]
        if any(word in combined_keywords.lower() for word in query_lower.split() if len(word) > 3):
            results.append(f"[{doc['source']}]: {doc['content']}")
    return results


# ==============================================================================
# Step 3: Define Structured Grading Schema
# ==============================================================================

class DocGrade(BaseModel):
    """Evaluation result for a single document's relevance to the user query."""
    relevance: Literal["yes", "no"] = Field(
        description="Choose 'yes' if the document is highly relevant to the query and contains information that helps answer it. Choose 'no' if the document is about a different product version, outdated, or completely off-topic."
    )
    reasoning: str = Field(description="Explanation of the grading decision.")

structured_grader = eval_model.with_structured_output(DocGrade)


# ==============================================================================
# Step 4: Define Graph State
# ==============================================================================

class CRAGState(MessagesState):
    """State containing retrieved documents, grading status, and query updates."""
    retrieved_docs: List[str]  # Active context list
    needs_web_search: bool     # Flag triggering corrective web search
    transformed_query: str     # Optimized query for web search
    generation: str            # Final compiled response


# ==============================================================================
# Step 5: Implement Graph Nodes
# ==============================================================================

def retrieve_local_node(state: CRAGState) -> dict:
    """Queries local vector database for matching records."""
    user_query = state["messages"][-1].content
    print(f"\n[1. Retriever] Searching local Vector DB for: '{user_query}'")
    
    docs = retrieve_local_faq(user_query)
    print(f"   Retrieved {len(docs)} document candidate(s).")
    
    return {"retrieved_docs": docs, "needs_web_search": False}


def evaluate_documents_node(state: CRAGState) -> dict:
    """Grades retrieved documents. Sets search flag if documents are irrelevant."""
    user_query = state["messages"][-1].content
    docs = state.get("retrieved_docs", [])
    
    print("\n[2. Evaluator] Grading document relevance...")
    
    if not docs:
        print("   No documents retrieved. Marking search fallback immediately.")
        return {"needs_web_search": True}
        
    valid_docs = []
    needs_search = False
    
    for doc in docs:
        prompt = (
            "Evaluate whether this document contains relevant details to answer the query. "
            "If the query asks about a specific product/version (e.g. SmartPlug Pro) and the "
            "document only discusses a different version (e.g. SmartPlug v1 or v2), mark relevance='no'.\n\n"
            f"User Query: {user_query}\n"
            f"Document:\n{doc}"
        )
        grade = structured_grader.invoke([HumanMessage(content=prompt)])
        
        if grade.relevance == "yes":
            print(f"   [Doc Relevant: YES] Details: {grade.reasoning}")
            valid_docs.append(doc)
        else:
            print(f"   [Doc Relevant: NO] Details: {grade.reasoning}")
            needs_search = True  # Trigger search fallback if any doc fails
            
    # If all docs were scored irrelevant, clear list to prevent polluted generation
    return {
        "retrieved_docs": valid_docs,
        "needs_web_search": needs_search or (len(valid_docs) == 0)
    }


def transform_query_node(state: CRAGState) -> dict:
    """Reformulates query to expand web search accuracy."""
    user_query = state["messages"][-1].content
    print(f"\n[3. Query Transformer] Optimizing query for external web search...")
    
    prompt = (
        "You are a search query optimizer. The user's query failed to find relevant results "
        "in our internal documentation. Rewrite the query to optimize it for a public search engine. "
        "Focus on search-friendly keywords, product brands, and specific features.\n\n"
        f"Original Query: {user_query}"
    )
    
    rewritten = transform_model.invoke([HumanMessage(content=prompt)])
    print(f"   Optimized Query: '{rewritten.content}'")
    
    return {"transformed_query": rewritten.content}


def execute_web_search_node(state: CRAGState) -> dict:
    """Retrieves corrective/supplementary context from the web search API."""
    query = state.get("transformed_query") or state["messages"][-1].content
    docs = state.get("retrieved_docs", [])
    
    print(f"\n[4. Web Search] Invoking fallback search API...")
    web_results = execute_external_web_search(query)
    
    print(f"   Found {len(web_results)} relevant web result(s).")
    
    # Merge local valid context with the new web context
    merged_docs = docs + web_results
    return {"retrieved_docs": merged_docs}


def generate_response_node(state: CRAGState) -> dict:
    """Compiles the final, factual response based on corrected context."""
    user_query = state["messages"][-1].content
    context = "\n\n".join(state.get("retrieved_docs", []))
    
    print("\n[5. Generator] Constructing finalized response...")
    
    prompt = (
        "You are ElectroNova Technical Support. Answer the user query using only the "
        "provided context. If the answer cannot be determined from the context, "
        "state that clearly. Do not make assumptions.\n\n"
        f"Context:\n{context if context else 'No document context available.'}\n\n"
        f"User Query: {user_query}"
    )
    
    response = gen_model.invoke([HumanMessage(content=prompt)])
    
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content, name="TechnicalSupport")]
    }


# ==============================================================================
# Step 6: Define Conditional Routers (State Machine Logic)
# ==============================================================================

def route_evaluator_decision(state: CRAGState) -> str:
    """Routes based on the document evaluation relevance flags."""
    if state["needs_web_search"]:
        return "transform_query"
    return "generate_response"


# Assemble the graph
workflow = StateGraph(CRAGState)

# Add Nodes
workflow.add_node("retrieve_local", retrieve_local_node)
workflow.add_node("evaluate_documents", evaluate_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("execute_web_search", execute_web_search_node)
workflow.add_node("generate_response", generate_response_node)

# Wire Edges
workflow.add_edge(START, "retrieve_local")
workflow.add_edge("retrieve_local", "evaluate_documents")

workflow.add_conditional_edges(
    "evaluate_documents",
    route_evaluator_decision,
    {
        "transform_query": "transform_query",
        "generate_response": "generate_response"
    }
)

workflow.add_edge("transform_query", "execute_web_search")
workflow.add_edge("execute_web_search", "generate_response")
workflow.add_edge("generate_response", END)

# Compile
compiled_crag = workflow.compile()


# ==============================================================================
# Step 7: Execution and Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CORRECTIVE RAG (CRAG) SUPPORT ENGINE - Start")
    print("=" * 80)
    
    # Test Cases:
    # 1. Matches local vector DB (Direct Gen)
    # 2. Out-of-Distribution product (Triggers CRAG search correction)
    
    queries = [
        "What features does the SmartPlug v2 have and how much does it cost?",
        "Tell me about the new SmartPlug Pro features and its pricing details."
    ]
    
    for idx, query in enumerate(queries, 1):
        print(f"\n\n[TEST CASE #{idx}] Query: \"{query}\"")
        print("-" * 80)
        
        inputs = {
            "messages": [HumanMessage(content=query)],
            "retrieved_docs": [],
            "needs_web_search": False,
            "transformed_query": "",
            "generation": ""
        }
        
        result = compiled_crag.invoke(inputs, {"recursion_limit": 20})
        
        print("\n[RESULT]")
        print(f"Final Output:\n{result['generation']}")
        print("=" * 80)
