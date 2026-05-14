"""
Supervisor Agent — the coordinator.

Reads the incoming scenario, decides which specialist agents to invoke and
in what order, then compiles their findings into the final recommendation.

UPGRADE: Uses Pydantic structured output via with_structured_output() rather
than regex-parsed JSON. This is the production-standard pattern — LangChain
converts the schema to JSON Schema, includes it in the prompt, validates the
response, and retries automatically on validation failure.
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from src.state.state import AgentState


# ─────────────────────────────────────────────────────────────────────────
# Structured output schemas
# ─────────────────────────────────────────────────────────────────────────

AgentName = Literal[
    "diagnostician",
    "interference_analyst",
    "capacity_planner",
    "policy_writer",
]


class RoutingDecision(BaseModel):
    """The supervisor's routing plan for a scenario."""
    routing_plan: list[AgentName] = Field(
        description=(
            "Ordered list of specialist agents to invoke. Always start with "
            "'diagnostician' (it identifies the cause). End with 'policy_writer' "
            "(it packages the A1 policy). Include 'interference_analyst' if the "
            "scenario mentions interference or coupling. Include 'capacity_planner' "
            "if the scenario mentions throughput, congestion, or capacity."
        )
    )
    reasoning: str = Field(
        description="One-sentence justification for the chosen routing plan."
    )


# ─────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────

SUPERVISOR_ROUTING_SYSTEM = """You are the Supervisor Agent in a multi-agent RF optimisation system.

Given a scenario, decide which specialist agents to invoke and in what order.

Available specialists:
- diagnostician: identifies what's wrong with a degraded cell (always invoke first)
- interference_analyst: identifies the dominant aggressor cell when interference is suspected
- capacity_planner: decides opex vs capex; computes NPV when capacity is in question
- policy_writer: packages findings into an O-RAN A1 policy (always invoke last)

Always include diagnostician first and policy_writer last. The middle agents depend on
the scenario type:
- Pure capacity issue: diagnostician -> capacity_planner -> policy_writer
- Pure interference issue: diagnostician -> interference_analyst -> policy_writer
- Mixed: diagnostician -> interference_analyst -> capacity_planner -> policy_writer
- Pure mobility: diagnostician -> policy_writer (mobility is handled inside diagnostician)
"""

SUPERVISOR_COMPILE_SYSTEM = """You are the Supervisor Agent compiling the final recommendation
for the RF planner. The specialist agents have completed their analysis. Your job is to
write a 4-6 sentence recommendation that:

1. Leads with the verdict (what the planner needs to know)
2. Cites specific numbers from the specialists' findings
3. States the recommended action and its expected impact
4. Notes confidence level and any caveats

Be specific, concise, and honest about uncertainty. Write for an experienced RF engineer."""


# ─────────────────────────────────────────────────────────────────────────
# Node factories
# ─────────────────────────────────────────────────────────────────────────

def make_supervisor_router(llm: ChatAnthropic):
    """Returns a node function that routes the scenario using structured output."""
    structured_llm = llm.with_structured_output(RoutingDecision)

    def route(state: AgentState) -> dict:
        scenario = state["scenario"]

        prompt = f"""Scenario received:

Type: {scenario.get('type')}
Description: {scenario.get('description')}
Affected cell: {scenario.get('affected_cell')}
Symptoms: {scenario.get('symptoms')}

Decide the routing plan."""

        # Structured output — guaranteed to return a valid RoutingDecision
        decision: RoutingDecision = structured_llm.invoke([
            {"role": "system", "content": SUPERVISOR_ROUTING_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        print(f"\n[SUPERVISOR] Scenario received: {scenario.get('type')}")
        print(f"[SUPERVISOR] Routing plan: {' -> '.join(decision.routing_plan)}")
        print(f"[SUPERVISOR] Reasoning: {decision.reasoning}")

        return {
            "messages": [AIMessage(content=f"Routing: {' -> '.join(decision.routing_plan)}")],
            "routing_plan": decision.routing_plan,
            "next_agent": decision.routing_plan[0] if decision.routing_plan else "supervisor_compile",
        }

    return route


def make_supervisor_compiler(llm: ChatAnthropic):
    """Returns a node function that compiles the final recommendation."""

    def compile_final(state: AgentState) -> dict:
        diag = state.get("diagnostic_findings", {})
        interf = state.get("interference_findings", {})
        cap = state.get("capacity_findings", {})
        policy = state.get("policy_output", {})

        prompt = f"""All specialist agents have reported. Compile the final recommendation.

Diagnostician findings:
{diag.get('diagnosis', 'not invoked')}

Interference analyst findings:
{interf.get('analysis', 'not invoked')}

Capacity planner findings:
{cap.get('verdict', 'not invoked')}

Generated A1 policy summary:
{policy.get('summary', 'not generated')}

Write the 4-6 sentence recommendation now."""

        response = llm.invoke([
            {"role": "system", "content": SUPERVISOR_COMPILE_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        recommendation = response.content if isinstance(response.content, str) else str(response.content)
        print(f"\n[SUPERVISOR] Final recommendation compiled.")

        return {
            "messages": [AIMessage(content=recommendation)],
            "final_recommendation": recommendation,
            "next_agent": "end",
        }

    return compile_final
