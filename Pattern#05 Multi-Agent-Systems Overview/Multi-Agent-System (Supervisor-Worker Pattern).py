"""
Multi-Agent Agentic Pattern — LangGraph Implementation
=======================================================

This script implements the Multi-Agent (Supervisor) pattern where:
1. A Supervisor decides which specialist agent to call next
2. A Researcher agent gathers information using web search
3. A Writer agent creates polished content from research
4. A Critic agent reviews quality and provides feedback

Architecture:
    START → Supervisor → (Researcher | Writer | Critic) → Supervisor → ... → END

Key Concepts:
- Role specialization: each agent has a focused system prompt and tools
- Supervisor routing: central orchestrator decides the workflow
- Scoped context: agents see shared history but focus on their role
- Built-in quality control via dedicated critic agent
"""

# ==============================================================================
# Step 1: Imports and Setup
# ==============================================================================

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Strong model for the supervisor — needs strategic reasoning
supervisor_model = ChatOpenAI(model='gpt-4o', temperature=0)

# Worker models — can use lighter models for cost efficiency
worker_model = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)


# ==============================================================================
# Step 2: Define State and Supervisor Schema
# ==============================================================================

class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""
    next: str = Field(
        description="The next agent to call. Must be one of: "
                    "'researcher', 'writer', 'critic', or 'FINISH'"
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen"
    )


class MultiAgentState(MessagesState):
    """
    State that flows through the multi-agent workflow.

    Fields:
        messages: Shared conversation history (inherited from MessagesState)
        next_agent: The supervisor's routing decision
    """
    next_agent: str


# ==============================================================================
# Step 3: Define the Supervisor Node
# ==============================================================================

supervisor_with_structure = supervisor_model.with_structured_output(SupervisorDecision)

TEAM_MEMBERS = ["researcher", "writer", "critic"]

def supervisor_node(state: MultiAgentState) -> dict:
    """
    Central supervisor that orchestrates the multi-agent workflow.

    Reads the full message history and decides:
    - Which specialist agent to call next
    - Or whether the task is complete (FINISH)
    """
    system_prompt = f"""You are a team supervisor managing a content production team.
Your team members are: {', '.join(TEAM_MEMBERS)}.

Each team member's specialty:
- **researcher**: Searches the web and gathers factual information. 
  Call this agent FIRST to gather raw material.
- **writer**: Takes research and creates well-structured, engaging content.
  Call this agent AFTER the researcher has gathered information.
- **critic**: Reviews written content for quality, accuracy, and completeness.
  Call this agent AFTER the writer has produced a draft.

Your workflow should generally follow: researcher → writer → critic → FINISH
However, if the critic finds significant issues, route back to the writer.

Rules:
- Call each agent at most 2 times to prevent infinite loops
- After the critic approves (or on the second review), select FINISH
- When selecting FINISH, the last substantial content in the conversation
  IS the final deliverable — do NOT regenerate it
- Consider the FULL conversation history when making decisions
"""

    messages = state["messages"]

    response = supervisor_with_structure.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )

    print(f"\n{'='*60}")
    print(f"🧑‍💼 SUPERVISOR → {response.next}")
    print(f"   Reason: {response.reasoning}")
    print(f"{'='*60}")

    return {"next_agent": response.next}


# ==============================================================================
# Step 4: Define the Specialist Agent Nodes
# ==============================================================================

# ── Researcher Agent ──────────────────────────────────────
search_tool = TavilySearchResults(max_results=3)

researcher_agent = create_react_agent(
    worker_model,
    tools=[search_tool],
    prompt=(
        "You are an expert researcher. Your job is to gather comprehensive, "
        "factual information on the given topic using web search.\n\n"
        "Guidelines:\n"
        "- Search for multiple aspects of the topic\n"
        "- Include specific facts, statistics, and recent developments\n"
        "- Cite your sources when possible\n"
        "- Organize findings clearly with headers/bullet points\n"
        "- Focus on accuracy — do NOT fabricate information"
    ),
)

def researcher_node(state: MultiAgentState) -> dict:
    """Research agent: gathers factual information using web search."""
    result = researcher_agent.invoke(state)
    last_message = result["messages"][-1]

    print(f"\n{'─'*60}")
    print(f"🔬 RESEARCHER completed")
    print(f"   Output preview: {last_message.content[:200]}...")
    print(f"{'─'*60}")

    return {
        "messages": [
            AIMessage(
                content=f"[RESEARCHER REPORT]\n\n{last_message.content}",
                name="researcher"
            )
        ]
    }


# ── Writer Agent ──────────────────────────────────────────
writer_agent = create_react_agent(
    worker_model,
    tools=[],
    prompt=(
        "You are an expert content writer. Your job is to take research "
        "findings and create polished, engaging content.\n\n"
        "Guidelines:\n"
        "- Use the research provided in the conversation to write factual content\n"
        "- Create a clear structure: introduction, body sections, conclusion\n"
        "- Write in an engaging, accessible style for a general audience\n"
        "- Include specific examples and data points from the research\n"
        "- If you received feedback from the critic, address ALL their points\n"
        "- Aim for 500-800 words\n"
        "- Do NOT make up facts — use only what the researcher provided"
    ),
)

def writer_node(state: MultiAgentState) -> dict:
    """Writer agent: creates polished content from research findings."""
    result = writer_agent.invoke(state)
    last_message = result["messages"][-1]

    print(f"\n{'─'*60}")
    print(f"✍️  WRITER completed")
    print(f"   Output preview: {last_message.content[:200]}...")
    print(f"{'─'*60}")

    return {
        "messages": [
            AIMessage(
                content=f"[WRITER DRAFT]\n\n{last_message.content}",
                name="writer"
            )
        ]
    }


# ── Critic Agent ──────────────────────────────────────────
critic_agent = create_react_agent(
    worker_model,
    tools=[],
    prompt=(
        "You are an expert content critic and editor. Your job is to "
        "review content for quality, accuracy, and completeness.\n\n"
        "Evaluate against these criteria:\n"
        "1. **Accuracy**: Are facts correctly represented from the research?\n"
        "2. **Structure**: Is the content well-organized with clear flow?\n"
        "3. **Engagement**: Is the writing compelling and readable?\n"
        "4. **Completeness**: Does it cover all key aspects of the topic?\n"
        "5. **Clarity**: Is the language clear and jargon-free?\n\n"
        "Provide:\n"
        "- A quality verdict: APPROVE or NEEDS_REVISION\n"
        "- Specific strengths\n"
        "- Specific weaknesses (if any)\n"
        "- Concrete suggestions for improvement (if NEEDS_REVISION)"
    ),
)

def critic_node(state: MultiAgentState) -> dict:
    """Critic agent: reviews content quality and provides feedback."""
    result = critic_agent.invoke(state)
    last_message = result["messages"][-1]

    print(f"\n{'─'*60}")
    print(f"🔍 CRITIC completed")
    print(f"   Output preview: {last_message.content[:200]}...")
    print(f"{'─'*60}")

    return {
        "messages": [
            AIMessage(
                content=f"[CRITIC REVIEW]\n\n{last_message.content}",
                name="critic"
            )
        ]
    }


# ==============================================================================
# Step 5: Define the Router (Conditional Edge)
# ==============================================================================

def route_to_agent(state: MultiAgentState) -> Literal[
    "researcher", "writer", "critic", "__end__"
]:
    """
    Route to the next agent based on the supervisor's decision.
    Maps supervisor choices to graph node names.
    """
    next_agent = state.get("next_agent", "FINISH")

    if next_agent == "FINISH":
        return "__end__"

    return next_agent


# ==============================================================================
# Step 6: Build the Graph
# ==============================================================================

# Create the graph
graph = StateGraph(MultiAgentState)

# Add nodes
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)

# Add edges
graph.add_edge(START, "supervisor")             # Start → Supervisor

graph.add_conditional_edges(                    # Supervisor → (Agent or End)
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "writer": "writer",
        "critic": "critic",
        "__end__": END,
    }
)

# All agents report back to the supervisor
graph.add_edge("researcher", "supervisor")      # Researcher → Supervisor
graph.add_edge("writer", "supervisor")          # Writer → Supervisor
graph.add_edge("critic", "supervisor")          # Critic → Supervisor

# Compile the workflow
workflow = graph.compile()


# ==============================================================================
# Step 7: Run It
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 MULTI-AGENT CONTENT TEAM — Starting")
    print("=" * 60)

    # Invoke the workflow with a content production request
    result = workflow.invoke({
        "messages": [
            HumanMessage(
                content="Write a comprehensive blog post about the current state "
                        "of quantum computing: recent breakthroughs, practical "
                        "applications, and what to expect in the next 5 years."
            )
        ]
    })

    # Print the final output (find the last writer message)
    print("\n" + "=" * 60)
    print("🎯 FINAL OUTPUT")
    print("=" * 60)

    for msg in reversed(result["messages"]):
        if hasattr(msg, "name") and msg.name == "writer":
            print(msg.content)
            break

    # Print collaboration summary
    print("\n" + "=" * 60)
    print("📊 COLLABORATION SUMMARY")
    print("=" * 60)

    agent_counts = {}
    for msg in result["messages"]:
        agent = getattr(msg, "name", msg.type)
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    print(f"  Total messages exchanged: {len(result['messages'])}")
    for agent, count in agent_counts.items():
        print(f"  {agent}: {count} message(s)")

    # Show agent conversation trail
    print("\n" + "=" * 60)
    print("🔍 AGENT CONVERSATION TRAIL")
    print("=" * 60)

    for i, msg in enumerate(result["messages"]):
        agent = getattr(msg, "name", msg.type)
        preview = msg.content[:150] if msg.content else "(no content)"
        print(f"\n  Message {i + 1} [{agent}]: {preview}")
        if len(msg.content or "") > 150:
            print("  ...")
