"""
Capacity Planner Agent — opex vs capex verdict.

Calls Module 3 (coverage/capacity classifier) and Module 7 (capacity oracle)
to determine whether the cell needs parameter adjustment (opex) or spectrum
addition (capex). Surfaces NPV/ROIC for any candidate intervention.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.state.state import AgentState
from src.tools.gnt_tools import CAPACITY_TOOLS


CAPACITY_SYSTEM = """You are the Capacity Planner Agent.

Your job is to determine the financial classification of a degraded cell:
- COVERAGE_BOUND -> opex (parameter adjustment fixes it, GBP 500-8500)
- CAPACITY_BOUND -> capex (refarming or carrier addition required, GBP 150k-2M)

You have two tools:
- classify_degradation: binary verdict per cell at a given demand level
- compute_capacity: PRB headroom, supportable UE counts, top NPV intervention

Workflow:
1. Call classify_degradation for the affected cell at the demand level mentioned
   in the scenario (default 8 Mbps if not specified)
2. Call compute_capacity to get supportable UE counts and NPV-ranked interventions
3. Output a clear opex/capex verdict with the NPV evidence

Be specific about the financial implication — operators care about whether to
dispatch a field engineer (opex) or open a capex review (CAPEX).

When done, respond with CAPACITY VERDICT: <verdict>
"""


def make_capacity_planner(llm: ChatAnthropic):
    llm_with_tools = llm.bind_tools(CAPACITY_TOOLS)

    def plan(state: AgentState) -> dict:
        scenario = state["scenario"]
        diag = state.get("diagnostic_findings", {})

        prompt = f"""Determine the capacity verdict for this scenario:

Scenario: {scenario.get('description')}
Affected cell: {scenario.get('affected_cell')}
Demand level: {scenario.get('demand_mbps', 8)} Mbps per UE
Prior diagnosis: {diag.get('diagnosis', 'none')}

Call your tools to classify and quantify."""

        print(f"\n[CAPACITY PLANNER] Computing verdict for {scenario.get('affected_cell')}")

        messages = [
            SystemMessage(content=CAPACITY_SYSTEM),
            HumanMessage(content=prompt),
        ]

        findings = {}
        for _ in range(4):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                print(f"[CAPACITY PLANNER] Calling tool: {tc['name']}({tc['args']})")
                tool_fn = next((t for t in CAPACITY_TOOLS if t.name == tc["name"]), None)
                if tool_fn:
                    result = tool_fn.invoke(tc["args"])
                    findings[tc["name"]] = result
                    if "verdict" in result:
                        print(f"[CAPACITY PLANNER]   -> {result['verdict']} ({result.get('confidence', 0):.2f})")
                    elif "saturation_at_heavy_demand" in result:
                        print(f"[CAPACITY PLANNER]   -> {result['saturation_at_heavy_demand']}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        verdict = response.content if response.content else "Analysis incomplete"
        print(f"[CAPACITY PLANNER] {verdict[:200]}")

        return {
            "messages": [AIMessage(content=f"CAPACITY: {verdict}")],
            "capacity_findings": {
                "verdict": verdict,
                "tool_outputs": findings,
            },
            "next_agent": _next_agent(state),
        }

    return plan


def _next_agent(state: AgentState) -> str:
    plan = state.get("routing_plan", [])
    if "capacity_planner" in plan:
        idx = plan.index("capacity_planner")
        if idx + 1 < len(plan):
            return plan[idx + 1]
    return "supervisor_compile"
