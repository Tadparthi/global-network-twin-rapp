"""
LangGraph state machine — the orchestration layer.

Wires the supervisor and four specialist agents into a directed graph.
Each agent reads and writes shared AgentState; routing is decided by
the supervisor and stored in state["next_agent"].

HUMAN-IN-THE-LOOP: When `human_in_the_loop=True` is passed to build_graph(),
the compiled graph uses a MemorySaver checkpointer and pauses before the
policy_writer node. This lets an RF planner review the diagnostic findings
and approve before any A1 policy is generated — the production pattern for
operator-supervised intervention workflows.

In production you'd replace MemorySaver with SqliteSaver or PostgresSaver
for persistence across process restarts. MemorySaver keeps state in-process,
which is fine for a portfolio demo but lost on restart.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from src.state.state import AgentState
from src.agents.supervisor import make_supervisor_router, make_supervisor_compiler
from src.agents.diagnostician import make_diagnostician
from src.agents.interference import make_interference_analyst
from src.agents.capacity import make_capacity_planner
from src.agents.policy_writer import make_policy_writer


def build_graph(
    model_name: str = "claude-sonnet-4-5",
    human_in_the_loop: bool = False,
) -> StateGraph:
    """
    Build the multi-agent graph.

    Args:
        model_name: Anthropic model identifier
        human_in_the_loop: If True, compile with a MemorySaver checkpointer
            and interrupt before the policy_writer node. The caller must then
            invoke twice: first to run up to the interrupt, then again with
            input=None to resume after human approval.

    Returns:
        Compiled LangGraph ready to invoke with a scenario.
    """
    llm = ChatAnthropic(model=model_name, max_tokens=2048, temperature=0)

    # Build node functions, each closing over the LLM
    supervisor_router = make_supervisor_router(llm)
    supervisor_compiler = make_supervisor_compiler(llm)
    diagnostician = make_diagnostician(llm)
    interference_analyst = make_interference_analyst(llm)
    capacity_planner = make_capacity_planner(llm)
    policy_writer = make_policy_writer(llm)

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("supervisor_route", supervisor_router)
    graph.add_node("diagnostician", diagnostician)
    graph.add_node("interference_analyst", interference_analyst)
    graph.add_node("capacity_planner", capacity_planner)
    graph.add_node("policy_writer", policy_writer)
    graph.add_node("supervisor_compile", supervisor_compiler)

    # Entry point — supervisor routes first
    graph.set_entry_point("supervisor_route")

    # Conditional routing — supervisor decides what runs next
    def route_from(state: AgentState) -> str:
        return state.get("next_agent", "supervisor_compile")

    graph.add_conditional_edges(
        "supervisor_route",
        route_from,
        {
            "diagnostician": "diagnostician",
            "interference_analyst": "interference_analyst",
            "capacity_planner": "capacity_planner",
            "policy_writer": "policy_writer",
            "supervisor_compile": "supervisor_compile",
            "end": END,
        },
    )

    for agent_name in ["diagnostician", "interference_analyst", "capacity_planner", "policy_writer"]:
        graph.add_conditional_edges(
            agent_name,
            route_from,
            {
                "diagnostician": "diagnostician",
                "interference_analyst": "interference_analyst",
                "capacity_planner": "capacity_planner",
                "policy_writer": "policy_writer",
                "supervisor_compile": "supervisor_compile",
                "end": END,
            },
        )

    # Final compilation goes to END
    graph.add_edge("supervisor_compile", END)

    # Compile with checkpointer + interrupt if human-in-the-loop is requested
    if human_in_the_loop:
        checkpointer = MemorySaver()
        return graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["policy_writer"],
        )
    return graph.compile()
