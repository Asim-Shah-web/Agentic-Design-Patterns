"""
Plan-and-Execute Agentic Pattern — LangGraph Implementation
============================================================

This script implements the Plan-and-Execute pattern where:
1. A Planner decomposes a complex goal into ordered sub-tasks
2. An Executor carries out each sub-task using tools (web search)
3. A Replanner evaluates progress and adapts the plan as needed

Architecture:
    START → Planner → Executor → Replanner → (Executor | END)

Key Concepts:
- Separation of planning (strategic) from execution (tactical)
- Model tiering: strong model for planning, lighter model for execution
- Adaptive replanning based on execution results
- Full audit trail of all steps and results
"""

# ==============================================================================
# Step 1: Imports and Setup
# ==============================================================================

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator

load_dotenv()

# Strong model for planning — needs to reason about the big picture
planner_model = ChatOpenAI(model='gpt-4o', temperature=0)

# Lighter model for execution — just needs to follow instructions + use tools
executor_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)


# ==============================================================================
# Step 2: Define Structured Output Schemas and State
# ==============================================================================

class Plan(BaseModel):
    """Ordered list of steps to achieve the user's goal."""
    steps: list[str] = Field(
        description="A list of steps to follow, in order. "
                    "Each step should be a clear, actionable task."
    )


class ReplanOutput(BaseModel):
    """Output of the replanner — either an updated plan or a final response."""
    updated_plan: list[str] = Field(
        default=None,
        description="Updated list of remaining steps (if more work needed)"
    )
    response: str = Field(
        default=None,
        description="Final response to the user (if goal is achieved)"
    )


class PlanExecuteState(TypedDict):
    """
    State that flows through the entire Plan-and-Execute workflow.
    
    Fields:
        input: The original user objective/question
        plan: Current list of sub-tasks (modified by planner and replanner)
        past_steps: Accumulated history of (task, result) pairs — uses reducer
        response: Final synthesized answer (set when goal is achieved)
    """
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str


# ==============================================================================
# Step 3: Define the Planner Node
# ==============================================================================

planner_with_structure = planner_model.with_structured_output(Plan)

def planner_node(state: PlanExecuteState) -> dict:
    """
    Decompose the user's objective into a structured, sequential plan.
    
    This is called ONCE at the start of the workflow. The planner analyzes
    the full goal and creates a roadmap of sub-tasks.
    """
    objective = state["input"]
    
    prompt = f"""You are an expert planning agent. Your job is to break down 
a complex objective into a clear, sequential plan of actionable steps.

Rules for creating the plan:
- Each step should be a SINGLE, specific, actionable task
- Steps should be in logical order (dependencies first)
- Each step should be self-contained enough for a worker to execute independently
- Include a final step to synthesize/compile the results into a final answer
- Keep the plan concise: 3-6 steps for most tasks (avoid over-decomposition)
- Consider what tools are available (web search) when planning steps

User's Objective: {objective}

Create the plan now:"""
    
    plan = planner_with_structure.invoke(prompt)
    
    print(f"\n{'='*60}")
    print(f"📋 PLAN CREATED — {len(plan.steps)} steps:")
    print(f"{'='*60}")
    for i, step in enumerate(plan.steps, 1):
        print(f"   {i}. {step}")
    print()
    
    return {"plan": plan.steps}


# ==============================================================================
# Step 4: Define the Executor Node
# ==============================================================================

# Set up tools for the executor
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# Create a ReAct agent as the executor — it can reason + use tools
executor_agent = create_react_agent(executor_model, tools)

def executor_node(state: PlanExecuteState) -> dict:
    """
    Execute the NEXT step in the plan using available tools.
    
    Takes the first step from the current plan, runs a ReAct agent
    to complete it, and records the result in past_steps.
    """
    plan = state["plan"]
    past_steps = state.get("past_steps", [])
    
    # The current task is the first step in the remaining plan
    current_task = plan[0]
    
    # Build context from previous steps so the executor has full history
    context = ""
    if past_steps:
        context = "\n\nContext from previously completed steps:\n"
        for step, result in past_steps:
            context += f"- Task: {step}\n  Result: {result}\n"
    
    # Run the ReAct agent on the current task
    agent_input = {
        "messages": [
            ("user", f"""You are a research assistant executing one specific task.
Complete the following task thoroughly and provide a detailed result.
{context}
Current Task: {current_task}

Execute this task now and provide your findings:""")
        ]
    }
    
    result = executor_agent.invoke(agent_input)
    
    # Extract the final message content
    agent_response = result["messages"][-1].content
    
    print(f"\n{'─'*60}")
    print(f"⚙️  EXECUTED: {current_task}")
    print(f"{'─'*60}")
    print(f"   Result preview: {agent_response[:200]}...")
    print()
    
    return {
        "past_steps": [(current_task, agent_response)],  # Appended via reducer
    }


# ==============================================================================
# Step 5: Define the Replanner Node
# ==============================================================================

replanner_with_structure = planner_model.with_structured_output(ReplanOutput)

def replanner_node(state: PlanExecuteState) -> dict:
    """
    Evaluate progress and decide: continue execution, adjust the plan, or finish.
    
    After each execution step, the replanner checks:
    1. Has the overall goal been achieved?
    2. Is the remaining plan still valid given new information?
    3. Should the plan be adjusted based on what was learned?
    """
    objective = state["input"]
    plan = state["plan"]
    past_steps = state.get("past_steps", [])
    
    # Format completed steps for the replanner
    past_steps_text = "\n".join(
        f"Step: {step}\nResult: {result}\n" for step, result in past_steps
    )
    
    # Format remaining plan (skip the step that was just executed)
    remaining_steps = plan[1:]
    remaining_text = "\n".join(
        f"{i+1}. {step}" for i, step in enumerate(remaining_steps)
    ) if remaining_steps else "No remaining steps."
    
    prompt = f"""You are a planning supervisor evaluating the progress of a 
research task. Based on the work completed so far, decide what to do next.

Original Objective: {objective}

Work Completed So Far:
{past_steps_text}

Remaining Plan:
{remaining_text}

Instructions:
- If the original objective has been FULLY answered by the completed work, 
  set 'response' to a comprehensive final answer that synthesizes ALL findings.
- If more work is needed, set 'updated_plan' to the list of remaining steps 
  (you may modify, add, or remove steps based on what you've learned).
- Do NOT repeat steps that have already been completed.
- The final response should be detailed and well-organized.

Decide now:"""
    
    output = replanner_with_structure.invoke(prompt)
    
    if output.response:
        print(f"\n{'='*60}")
        print(f"✅ GOAL ACHIEVED — Generating final response")
        print(f"{'='*60}\n")
        return {"response": output.response}
    else:
        print(f"\n🔄 REPLANNED: {len(output.updated_plan)} steps remaining")
        for i, step in enumerate(output.updated_plan, 1):
            print(f"   {i}. {step}")
        return {"plan": output.updated_plan}


# ==============================================================================
# Step 6: Define the Decision Edge
# ==============================================================================

def should_end(state: PlanExecuteState) -> str:
    """
    Decision gate: determines whether the workflow is complete.
    
    Routes to END if a final response has been generated,
    otherwise routes back to the executor for the next step.
    """
    if state.get("response"):
        return END
    return "executor"


# ==============================================================================
# Step 7: Build the Graph
# ==============================================================================

# Create the graph
graph = StateGraph(PlanExecuteState)

# Add nodes
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("replanner", replanner_node)

# Add edges
graph.add_edge(START, "planner")           # Start → Planner (runs once)
graph.add_edge("planner", "executor")      # Planner → Executor (begin executing)
graph.add_edge("executor", "replanner")    # Executor → Replanner (evaluate progress)
graph.add_conditional_edges(               # Replanner → (Executor OR End)
    "replanner",
    should_end,
    {
        "executor": "executor",
        END: END,
    }
)

# Compile the workflow
workflow = graph.compile()


# ==============================================================================
# Step 8: Run It
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 PLAN-AND-EXECUTE AGENT — Starting")
    print("=" * 60)
    
    # Invoke the workflow with a complex research question
    result = workflow.invoke({
        "input": "Compare the GDP growth rates of India, China, and the US "
                 "over the last 3 years and explain the key drivers behind "
                 "each country's performance.",
        "plan": [],
        "past_steps": [],
        "response": "",
    })
    
    # Print the final answer
    print("\n" + "=" * 60)
    print("🎯 FINAL ANSWER")
    print("=" * 60)
    print(result["response"])
    
    # Print execution summary
    print("\n" + "=" * 60)
    print("📊 EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total steps executed: {len(result['past_steps'])}")
    for i, (step, res) in enumerate(result["past_steps"], 1):
        print(f"\n  Step {i}: {step}")
        print(f"  Result: {res[:200]}...")
