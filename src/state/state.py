"""
Shared state schema passed between all agents in the LangGraph.

The state is the message bus: each agent reads what it needs from this
dict, runs its tools, and writes its findings back. The supervisor reads
the final state to compile the recommendation.
"""
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """
    State shared across the multi-agent graph.

    `messages` is append-only across all agents (LangGraph reduces with `add`).
    Other fields are written once by their owning agent and read by downstream agents.
    """
    # Message history accumulated across the run
    messages: Annotated[Sequence[BaseMessage], add]

    # Scenario input
    scenario: dict

    # Supervisor's routing decision — which agents to invoke and in what order
    routing_plan: list[str]

    # Findings from each specialist agent
    diagnostic_findings: dict       # from Diagnostician
    interference_findings: dict     # from Interference Analyst
    capacity_findings: dict         # from Capacity Planner
    policy_output: dict             # from Policy Writer

    # Final recommendation compiled by the supervisor
    final_recommendation: str

    # Tracks which agent is currently active — used by graph routing
    next_agent: str
