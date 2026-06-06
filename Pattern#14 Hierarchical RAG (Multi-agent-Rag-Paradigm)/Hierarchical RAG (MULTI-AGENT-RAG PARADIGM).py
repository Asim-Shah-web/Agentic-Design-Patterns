"""
Hierarchical Multi-Agent RAG Pattern - LangGraph Implementation
==============================================================

This script implements a Hierarchical (Orchestrator-Worker) Multi-Agent RAG pattern:
1. M&A Deal Supervisor (Orchestrator): Receives a complex due diligence request,
   decomposes it, dynamically routes to specialized worker sub-agents, and
   synthesizes a high-level fact-grounded investment briefing.
2. Financial RAG Agent (Worker): Specialized in balance sheets, revenues, and debt metrics.
3. IP & Technology RAG Agent (Worker): Specialized in patents, whitepapers, and hardware claims.
4. Legal & Compliance RAG Agent (Worker): Specialized in litigation, FTC antitrust, and audits.

Key Concepts:
- Dynamic Routing with Structured Output: The Supervisor uses Pydantic to select the next worker and draft sub-queries.
- Information Hiding: Workers run full ReAct retrieval loops locally, but only return a synthesized, high-level summary briefing back to the Supervisor graph. This prevents context bloat.
- Isolated Specialized Search Tools: Each worker agent interacts with its own separate database mock.
"""

import os
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Set up models with dual-provider support (OpenAI and Groq)
if os.environ.get("OPENAI_API_KEY"):
    print("[INFO] Detected OpenAI API Key. Running with OpenAI (gpt-4o & gpt-4o-mini)...")
    from langchain_openai import ChatOpenAI
    orchestrator_model = ChatOpenAI(model='gpt-4o', temperature=0)
    worker_model = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
elif os.environ.get("GROQ_API_KEY"):
    print("[INFO] Detected Groq API Key. Running with Groq (llama-3.3-70b-versatile & llama-3.1-8b-instant)...")
    from langchain_groq import ChatGroq
    # Llama 3.3 70B is an exceptional, fast model for planning and structured output
    orchestrator_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    # Llama 3.1 8B is perfect, fast, and lightweight for worker agents
    worker_model = ChatGroq(model='llama-3.1-8b-instant', temperature=0.2)
else:
    raise ValueError("Missing credentials. Please set either OPENAI_API_KEY or GROQ_API_KEY in your environment / .env file.")


# ==============================================================================
# Step 1: Define Mock Knowledge Bases (Corporate Databases)
# ==============================================================================

# 1. Financial Database (SEC 10-K / Balance Sheet Filings)
FINANCIAL_DB = [
    {
        "company": "QuantumTech Inc.",
        "year": 2025,
        "document": (
            "QuantumTech Inc. SEC Form 10-K (FY2025):\n"
            "- Total Revenue: $142.5 Million USD (up 28% year-over-year from $111.3 Million in 2024).\n"
            "- Gross Margin: 64.2% driven by enterprise quantum SaaS subscriptions.\n"
            "- Operating Income (EBITDA): $38.4 Million USD.\n"
            "- Cash and Equivalents: $45.2 Million USD.\n"
            "- Total Debt Liabilities: $12.0 Million USD (Long-term senior convertible notes).\n"
            "- Principal Risks: High R&D overhead ($52.0M spent in 2025) and reliance on specialized chip fabrication foundries."
        ),
        "source": "SEC-10K-FY2025"
    },
    {
        "company": "QuantumTech Inc.",
        "year": 2024,
        "document": (
            "QuantumTech Inc. SEC Form 10-K (FY2024):\n"
            "- Total Revenue: $111.3 Million USD.\n"
            "- EBITDA: $21.1 Million USD.\n"
            "- Cash and Equivalents: $22.4 Million USD.\n"
            "- Total Debt Liabilities: $15.5 Million USD."
        ),
        "source": "SEC-10K-FY2024"
    }
]

# 2. Intellectual Property Database (USPTO Patents & Publications)
PATENT_DB = [
    {
        "company": "QuantumTech Inc.",
        "patent_id": "US-11948271-B2",
        "title": "Silicon-integrated superconducting quantum key distribution transceiver",
        "abstract": (
            "Abstract: This invention describes a micro-chip scale hardware transceiver utilizing "
            "silicon photonics to emit entangled photon pairs at telecom wavelengths (1550nm). "
            "Integrates superconducting nanowire single-photon detectors (SNSPDs) on a single silicon substrate. "
            "Enables cryptographic key distribution secure against quantum computer Shor-algorithm decryption."
        ),
        "filing_date": "2024-03-12",
        "source": "USPTO-PAT-US-11948271"
    },
    {
        "company": "QuantumTech Inc.",
        "patent_id": "US-12053910-B1",
        "title": "Cryogenic thermal isolating packaging for multi-qubit processors",
        "abstract": (
            "Abstract: A packaging system comprising nested vacuum-insulated chambers designed to isolate "
            "a 128-qubit processor core from thermal electromagnetic radiation down to 10 milli-Kelvin. "
            "Uses gold-plated copper shielding and superconducting coaxial trace paths to minimize signal cross-talk."
        ),
        "filing_date": "2025-01-08",
        "source": "USPTO-PAT-US-12053910"
    }
]

# 3. Legal & Regulatory Database (Court Dockets & Compliance Records)
LEGAL_DB = [
    {
        "company": "QuantumTech Inc.",
        "case_id": "FTC-2025-A-928",
        "matter": "Antitrust & Monopoly Review regarding proposed merger with CryoSystems Ltd.",
        "status": "Ongoing / Under Review",
        "details": (
            "Details: FTC Bureau of Competition opened a preliminary investigation in October 2025 "
            "to evaluate whether QuantumTech's proposed $30M acquisition of CryoSystems Ltd. (sole manufacturer "
            "of cryogenic dilution refrigeration valves) constitutes a vertical monopoly in the quantum supply chain. "
            "QuantumTech has filed a response arguing alternative valve fabricators exist in the EU."
        ),
        "source": "FTC-Antitrust-Docket-928"
    },
    {
        "company": "QuantumTech Inc.",
        "case_id": "DEL-CH-10922-2025",
        "matter": "Patent Infringement Lawsuit filed by CyberSec Global Corp.",
        "status": "Active / Pre-trial discovery",
        "details": (
            "Details: CyberSec Global Corp filed a civil suit in Delaware Chancery Court in November 2025, "
            "claiming QuantumTech's silicon-integrated photonics transceiver (Patent US-11948271-B2) infringes "
            "upon CyberSec's core patent US-10822194 covering coherent optical ring resonators. "
            "QuantumTech legal counsel is preparing an invalidity defense, estimating a 75% probability of winning."
        ),
        "source": "Delaware-Chancery-Court-CH-10922"
    }
]


# ==============================================================================
# Step 2: Define Specialized Retrieval Tools
# ==============================================================================

@tool
def query_sec_financials(company: str, query: str) -> str:
    """Retrieves financial filings, revenues, EBITDA, balance sheets, and debt liabilities for a given company."""
    results = []
    for doc in FINANCIAL_DB:
        if company.lower() in doc["company"].lower() or company.lower() in doc["document"].lower():
            results.append(f"Source: [{doc['source']}]\n{doc['document']}")
    
    if not results:
        return f"No financial documents found matching '{company}'."
    return "\n\n---\n\n".join(results)

@tool
def query_ip_patents(company: str, query: str) -> str:
    """Retrieves patent abstracts, titles, and technical system details for a given company's IP holdings."""
    results = []
    for doc in PATENT_DB:
        if company.lower() in doc["company"].lower() or company.lower() in doc["title"].lower() or company.lower() in doc["abstract"].lower():
            results.append(f"Source: [{doc['source']}]\nPatent ID: {doc['patent_id']}\nTitle: {doc['title']}\n{doc['abstract']}")
            
    if not results:
        return f"No intellectual property found matching '{company}'."
    return "\n\n---\n\n".join(results)

@tool
def query_legal_compliance(company: str, query: str) -> str:
    """Retrieves lawsuits, regulatory compliance audits, FTC investigations, and court dockets for a given company."""
    results = []
    for doc in LEGAL_DB:
        if company.lower() in doc["company"].lower() or company.lower() in doc["matter"].lower() or company.lower() in doc["details"].lower():
            results.append(f"Source: [{doc['source']}]\nCase ID: {doc['case_id']}\nMatter: {doc['matter']}\nStatus: {doc['status']}\n{doc['details']}")
            
    if not results:
        return f"No legal/compliance cases found matching '{company}'."
    return "\n\n---\n\n".join(results)


# ==============================================================================
# Step 3: Build Specialized Retrieval Worker Agents
# ==============================================================================

# We compile separate ReAct agents for each domain. These are completely decoupled,
# meaning we can scale their tools, prompt structures, or models independently!
financial_agent = create_react_agent(
    worker_model,
    tools=[query_sec_financials],
    prompt=(
        "You are an expert Financial Auditor and SEC RAG specialist. Your task is to query "
        "the financial records for the requested company, extract revenues, margin, EBITDA, "
        "debt, and financial risks, and formulate a highly structured Financial Briefing. "
        "Always cite your sources exactly using [SEC-10K-FYXXXX] format."
    )
)

ip_agent = create_react_agent(
    worker_model,
    tools=[query_ip_patents],
    prompt=(
        "You are an expert Patent Analyst and Tech RAG specialist. Your task is to query "
        "patent databases for the requested company, outline their key hardware and optical claims, "
        "and explain the technical mechanics of their integrated circuits. "
        "Always cite your sources exactly using [USPTO-PAT-US-XXXXXX] format."
    )
)

legal_agent = create_react_agent(
    worker_model,
    tools=[query_legal_compliance],
    prompt=(
        "You are a Corporate Legal Counsel and Compliance RAG specialist. Your task is to query "
        "litigation dockets and FTC filings for the requested company. Summarize active lawsuits, "
        "antitrust issues, ongoing audits, and estimate their legal risks. "
        "Always cite your sources exactly using [Docket-ID] format."
    )
)


# ==============================================================================
# Step 4: Define Parent Graph State and Structured Supervisor Router
# ==============================================================================

class DiligenceState(MessagesState):
    """The central state for the Orchestrator graph."""
    # Tracks the supervisor's dynamic routing decisions
    next_agent: str
    sub_query: str
    
    # Store the synthesized reports from the individual workers
    # (Information Hiding: the Supervisor reviews these summaries, not the raw doc chunks)
    financial_report: str
    ip_report: str
    legal_report: str


# Pydantic schema for Supervisor's structured routing decisions
class RouterDecision(BaseModel):
    next: Literal["financial_worker", "ip_worker", "legal_worker", "FINISH"] = Field(
        description="Select the next domain specialist to consult. Select FINISH only when you have collected all necessary briefings."
    )
    sub_query: str = Field(
        description="The tailored query/question passed to the specialist worker. Keep it focused on their specific domain."
    )
    reasoning: str = Field(
        description="Explain why you are choosing this next step or why you have sufficient data to FINISH."
    )

# Instantiate structured router
structured_supervisor_router = orchestrator_model.with_structured_output(RouterDecision)


# ==============================================================================
# Step 5: Implement Graph Nodes (Supervisor & Worker Wrappers)
# ==============================================================================

def deal_supervisor(state: DiligenceState) -> dict:
    """The high-level orchestrator node that decomposes the task, manages routing, and plans next steps."""
    system_prompt = SystemMessage(content=(
        "You are the M&A Due Diligence Deal Supervisor (Orchestrator). You coordinate an audit of QuantumTech Inc. "
        "You manage three specialized retrieval worker agents:\n"
        "- financial_worker: Expert in balance sheets, revenues, cash, and EBITDA.\n"
        "- ip_worker: Expert in patent abstracts and technical hardware claims.\n"
        "- legal_worker: Expert in court lawsuits, FTC investigations, and regulatory actions.\n\n"
        "Your task is to analyze the target company across all three dimensions. "
        "Do not write or guess the facts yourself. Route requests to the workers one by one to gather facts. "
        "Formulate clear, specific queries in the 'sub_query' field.\n"
        "Once you have gathered reports from all necessary domains, select next='FINISH' to write your unified final report."
    ))
    
    # Call structured LLM
    decision = structured_supervisor_router.invoke([system_prompt] + state["messages"])
    
    print(f"\n[Supervisor] Routing to: {decision.next.upper()}")
    print(f"   Reasoning: {decision.reasoning}")
    if decision.next != "FINISH":
        print(f"   Formulated Query: '{decision.sub_query}'")
        
    return {
        "next_agent": decision.next,
        "sub_query": decision.sub_query,
        # Append supervisor's reasoning trace to state messages
        "messages": [AIMessage(content=f"[Supervisor Planning]: Routing to {decision.next}. Target search: {decision.sub_query}", name="DealSupervisor")]
    }

# -- Information Hiding Wrappers --------------------------
# Rather than adding raw retrieval results directly to the supervisor's context,
# these wrappers execute the worker agent, grab the final synthesized summary,
# store it in a dedicated State field, and return only the high-level summary to the supervisor's message log.

def run_financial_worker(state: DiligenceState) -> dict:
    query = state["sub_query"]
    print(f"   [Financial Worker] Consulting Financial Database for: '{query}'...")
    
    # Execute the react agent
    result = financial_agent.invoke({"messages": [HumanMessage(content=query)]})
    final_brief = result["messages"][-1].content
    
    return {
        "messages": [AIMessage(content=f"[FINANCIAL REPORT BRIEFING]:\n{final_brief}", name="FinancialWorker")],
        "financial_report": final_brief
    }

def run_ip_worker(state: DiligenceState) -> dict:
    query = state["sub_query"]
    print(f"   [IP Worker] Consulting Patent & Tech Database for: '{query}'...")
    
    # Execute the react agent
    result = ip_agent.invoke({"messages": [HumanMessage(content=query)]})
    final_brief = result["messages"][-1].content
    
    return {
        "messages": [AIMessage(content=f"[IP & PATENT REPORT BRIEFING]:\n{final_brief}", name="IPWorker")],
        "ip_report": final_brief
    }

def run_legal_worker(state: DiligenceState) -> dict:
    query = state["sub_query"]
    print(f"   [Legal Worker] Consulting Court Dockets for: '{query}'...")
    
    # Execute the react agent
    result = legal_agent.invoke({"messages": [HumanMessage(content=query)]})
    final_brief = result["messages"][-1].content
    
    return {
        "messages": [AIMessage(content=f"[LEGAL & COMPLIANCE BRIEFING]:\n{final_brief}", name="LegalWorker")],
        "legal_report": final_brief
    }

def synthesize_final_report(state: DiligenceState) -> dict:
    """Final node that compiles all retrieved domain briefings into a beautiful corporate due diligence summary."""
    print("\n[Supervisor] Compiling comprehensive investment audit...")
    
    prompt = (
        "You are the Deal Supervisor. You have gathered individual briefings from the Financial, IP, and Legal expert worker agents.\n"
        "Compile a comprehensive, beautifully structured M&A Due Diligence Audit Report for QuantumTech Inc.\n\n"
        f"1. FINANCIAL REPORT REPORT SUMMARY:\n{state['financial_report']}\n\n"
        f"2. INTELLECTUAL PROPERTY PATENT SUMMARY:\n{state['ip_report']}\n\n"
        f"3. LEGAL & COMPLIANCE SUMMARY:\n{state['legal_report']}\n\n"
        "Draft a cohesive report including an executive summary, a breakdown of findings for each department, "
        "an M&A risk assessment grade (e.g. Low, Medium, High Risk), and a final investment recommendation.\n"
        "Crucial Requirement: You must maintain all source citations (e.g., [SEC-10K-FY2025], [USPTO-PAT-US-XXXXXX]) exactly as provided by the workers."
    )
    
    response = orchestrator_model.invoke([SystemMessage(content=prompt)] + state["messages"])
    
    return {"messages": [AIMessage(content=response.content, name="DealSupervisor")]}


# ==============================================================================
# Step 6: Assemble and Compile the Graph
# ==============================================================================

def route_next(state: DiligenceState) -> str:
    """Conditional routing function."""
    if state["next_agent"] == "financial_worker":
        return "financial_worker"
    elif state["next_agent"] == "ip_worker":
        return "ip_worker"
    elif state["next_agent"] == "legal_worker":
        return "legal_worker"
    return "synthesize"

workflow = StateGraph(DiligenceState)

# Add Nodes
workflow.add_node("supervisor", deal_supervisor)
workflow.add_node("financial_worker", run_financial_worker)
workflow.add_node("ip_worker", run_ip_worker)
workflow.add_node("legal_worker", run_legal_worker)
workflow.add_node("synthesize", synthesize_final_report)

# Set edges
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "financial_worker": "financial_worker",
        "ip_worker": "ip_worker",
        "legal_worker": "legal_worker",
        "synthesize": "synthesize"
    }
)
workflow.add_edge("financial_worker", "supervisor")
workflow.add_edge("ip_worker", "supervisor")
workflow.add_edge("legal_worker", "supervisor")
workflow.add_edge("synthesize", END)

# Compile Graph
compiled_workflow = workflow.compile()


# ==============================================================================
# Step 7: Run the System
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HIERARCHICAL MULTI-AGENT RAG (M&A DUE DILIGENCE ENGINE) - Starting")
    print("=" * 80)
    
    # A complex user prompt requiring retrieval across multiple heterogeneous sources
    user_query = (
        "We are evaluating a potential M&A deal with QuantumTech Inc. "
        "Retrieve and audit their recent 2025 financial figures, check if their patent holdings "
        "support their silicon-photonics hardware claims, and audit their litigation dockets "
        "for active lawsuits or FTC antitrust problems. Assemble a cohesive investment review."
    )
    
    print(f"User Inquiry:\n\"{user_query}\"\n")
    
    # Run the workflow
    # We increase the recursion_limit because the graph loops back to the supervisor
    inputs = {"messages": [HumanMessage(content=user_query)]}
    result = compiled_workflow.invoke(inputs, {"recursion_limit": 50})
    
    print("\n" + "=" * 80)
    print("FINAL DUE DILIGENCE REPORT")
    print("=" * 80)
    print(result["messages"][-1].content)
    print("=" * 80)
    
    # Verify the "Information Hiding" benefit (Context window footprint)
    print("\nCONTEXT FOOTPRINT ANALYSIS (Information Hiding Demonstration)")
    print("=" * 80)
    print("This shows that the Supervisor only stored domain briefings, not the raw retrieved vector chunks:")
    
    total_chars = 0
    for idx, msg in enumerate(result["messages"]):
        name = getattr(msg, "name", None) or "User"
        role = "Orchestrator" if name == "DealSupervisor" else ("Worker" if "Worker" in name else "User")
        content_len = len(msg.content) if msg.content else 0
        total_chars += content_len
        print(f"[{idx+1:02d}] {name:<18} ({role:<12}): Message Length = {content_len:<4} characters")
        
    print(f"\nTotal characters kept in active message list: {total_chars}")
    print("=" * 80 + "\n")
