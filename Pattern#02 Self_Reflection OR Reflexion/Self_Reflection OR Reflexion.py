"""
Self-Reflection (Reflexion) Agentic Pattern — Built from Scratch with LangGraph
================================================================================

This script implements the Self-Reflection pattern:
  1. GENERATOR writes an essay on a given topic
  2. REFLECTOR critiques the essay with structured feedback (score, strengths, weaknesses, suggestions)
  3. CONTROLLER decides whether to accept (score >= 8) or loop back (up to 3 iterations max)

The generator receives the full previous draft + reflection feedback on each iteration,
enabling targeted improvements rather than starting from scratch.

Usage:
    python 16_self_reflection.py
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator

# ─────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ─────────────────────────────────────────────────────────
# Structured Output Schema for the Reflector
# ─────────────────────────────────────────────────────────

class ReflectionOutput(BaseModel):
    """Structured reflection output with quality scores and feedback."""

    strengths: str = Field(description="What the essay does well")
    weaknesses: str = Field(
        description="Specific problems and areas for improvement"
    )
    suggestions: str = Field(
        description="Concrete, actionable suggestions for the next draft"
    )
    score: int = Field(
        description=(
            "Overall quality score from 1-10. "
            "1-3: Poor, 4-5: Below average, 6-7: Good, "
            "8-9: Very good, 10: Excellent"
        ),
        ge=1,
        le=10,
    )


reflector_model = model.with_structured_output(ReflectionOutput)


# ─────────────────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────────────────

def add_to_list(existing: list, new: list) -> list:
    """Reducer that appends new items to the existing list."""
    return existing + new


class ReflectionState(TypedDict):
    topic: str  # The original user request
    draft: str  # Current essay draft
    reflection: str  # Latest reflection text
    score: int  # Latest quality score (1-10)
    drafts: Annotated[list[str], add_to_list]  # History of all drafts
    reflections: Annotated[list[str], add_to_list]  # History of all reflections
    iteration: Annotated[int, operator.add]  # Auto-incremented iteration counter


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 8


# ─────────────────────────────────────────────────────────
# Node: Generator
# ─────────────────────────────────────────────────────────

def generator_node(state: ReflectionState) -> dict:
    """
    Generate or refine an essay based on the topic and any previous reflection.

    First iteration:  Writes a fresh draft from the topic alone.
    Later iterations: Rewrites using the reflection feedback as guidance.
    """
    topic = state["topic"]
    iteration = state.get("iteration", 0)

    if iteration == 0:
        # ── FIRST DRAFT ──────────────────────────────────
        prompt = f"""You are an expert essay writer. Write a well-structured, 
insightful essay on the following topic.

Topic: {topic}

Requirements:
- Clear thesis statement in the introduction
- 3-4 body paragraphs with supporting evidence and examples
- Logical flow between paragraphs with smooth transitions
- A compelling conclusion that synthesizes the main points
- Engaging writing style appropriate for an educated audience
- Approximately 500-700 words

Write the essay now:"""

    else:
        # ── REFINEMENT (uses previous draft + reflection) ──
        previous_draft = state["draft"]
        reflection = state["reflection"]

        history_context = ""
        if len(state.get("drafts", [])) > 1:
            history_context = f"""
Previous iterations: {len(state['drafts'])}
Note: You have already revised this essay {len(state['drafts']) - 1} time(s). 
Focus on the LATEST feedback and avoid re-introducing issues from earlier drafts.
"""

        prompt = f"""You are an expert essay writer revising your work based on 
detailed feedback from a critic.

Original Topic: {topic}
{history_context}

Your Previous Draft:
---
{previous_draft}
---

Critic's Feedback:
---
{reflection}
---

Instructions:
- Address EVERY weakness and suggestion mentioned in the feedback
- Preserve the strengths that were noted
- Maintain or improve the overall structure
- Do NOT simply add disclaimers or meta-commentary about the changes
- Write the COMPLETE revised essay (not just the changed parts)

Write the improved essay now:"""

    response = model.invoke(prompt)
    new_draft = response.content

    return {
        "draft": new_draft,
        "drafts": [new_draft],  # Appended to history via reducer
        "iteration": 1,  # Incremented via operator.add reducer
    }


# ─────────────────────────────────────────────────────────
# Node: Reflector
# ─────────────────────────────────────────────────────────

def reflector_node(state: ReflectionState) -> dict:
    """
    Critically evaluate the current draft against quality criteria.

    Produces structured feedback with strengths, weaknesses,
    suggestions, and a numeric quality score (1-10).
    """
    topic = state["topic"]
    draft = state["draft"]
    iteration = state.get("iteration", 1)

    previous_reflections = ""
    if state.get("reflections"):
        previous_reflections = (
            "\nPrevious Reflections (check if earlier issues were fixed):\n---\n"
            + "\n".join(
                f"Iteration {i+1}: {r}"
                for i, r in enumerate(state["reflections"])
            )
            + "\n---\n"
        )

    prompt = f"""You are a demanding but fair essay critic and writing professor.
Evaluate the following essay draft rigorously against these criteria:

1. **Thesis & Argument** — Is there a clear, debatable thesis? Is it well-supported?
2. **Structure & Organization** — Logical flow? Clear paragraphs? Good transitions?
3. **Evidence & Depth** — Specific examples? Deep analysis vs. surface-level?
4. **Writing Quality** — Clarity? Engagement? Appropriate tone? Grammar?
5. **Completeness** — Does it fully address the topic? Any missing perspectives?

This is iteration {iteration} of the revision process.
{previous_reflections}

Topic: {topic}

Essay Draft:
---
{draft}
---

Be specific in your feedback. Point to exact sentences or paragraphs when 
discussing weaknesses. Your suggestions should be concrete and actionable.

IMPORTANT scoring guidelines:
- 1-3: Fundamental problems (no thesis, incoherent structure, major errors)
- 4-5: Below average (weak thesis, shallow analysis, poor organization)
- 6-7: Good but clearly improvable (decent structure, needs more depth/examples)
- 8-9: Very good (strong thesis, good evidence, minor issues only)
- 10: Exceptional (publishable quality, no meaningful improvements needed)"""

    reflection = reflector_model.invoke(prompt)

    reflection_text = f"""## Reflection (Iteration {iteration})

**Score: {reflection.score}/10**

### Strengths
{reflection.strengths}

### Weaknesses
{reflection.weaknesses}

### Suggestions for Improvement
{reflection.suggestions}"""

    return {
        "reflection": reflection_text,
        "reflections": [reflection_text],  # Appended to history via reducer
        "score": reflection.score,
    }


# ─────────────────────────────────────────────────────────
# Conditional Edge: Should Continue?
# ─────────────────────────────────────────────────────────

def should_continue(state: ReflectionState) -> Literal["generator", "__end__"]:
    """
    Decision gate: accept the draft or send it back for revision.

    Accepts if:
      - score >= QUALITY_THRESHOLD  (quality is good enough)
      - OR iteration >= MAX_ITERATIONS (safety valve)
    """
    score = state.get("score", 0)
    iteration = state.get("iteration", 0)

    if score >= QUALITY_THRESHOLD:
        print(
            f"\n✅ Draft accepted! Score: {score}/10 "
            f"after {iteration} iteration(s)"
        )
        return "__end__"

    if iteration >= MAX_ITERATIONS:
        print(
            f"\n⚠️  Max iterations ({MAX_ITERATIONS}) reached. "
            f"Final score: {score}/10"
        )
        return "__end__"

    print(
        f"\n🔄 Iteration {iteration}: Score {score}/10 "
        f"— sending back for revision..."
    )
    return "generator"


# ─────────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────────

graph = StateGraph(ReflectionState)

graph.add_node("generator", generator_node)
graph.add_node("reflector", reflector_node)

graph.add_edge(START, "generator")
graph.add_edge("generator", "reflector")
graph.add_conditional_edges("reflector", should_continue)

workflow = graph.compile()


# ─────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("🧠 SELF-REFLECTION ESSAY WRITER")
    print("=" * 80)

    result = workflow.invoke(
        {
            "topic": (
                "The impact of artificial intelligence on creative professions: "
                "threat or transformation?"
            ),
            "drafts": [],
            "reflections": [],
            "iteration": 0,
            "score": 0,
            "draft": "",
            "reflection": "",
        }
    )

    # ── Final Output ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("📝 FINAL ESSAY")
    print("=" * 80)
    print(result["draft"])

    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"  Final Score:       {result['score']}/10")
    print(f"  Total Iterations:  {result['iteration']}")
    print(f"  Drafts Generated:  {len(result['drafts'])}")

    # ── Reflection History ────────────────────────────────
    print("\n" + "=" * 80)
    print("🔍 REFLECTION HISTORY")
    print("=" * 80)

    for i, reflection in enumerate(result["reflections"]):
        print(f"\n{'─' * 40}")
        print(reflection)

    print(f"\n{'─' * 40}")
    print("Done!")
