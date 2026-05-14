"""
Interference Analyst Agent — identifies the dominant aggressor.

Calls Module 2 (coupling GNN) to determine which neighbour cell is causing
interference at the victim cell. The structural finding from this agent is
what prevents compensatory loss — adjusting the aggressor, not the symptom cell.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.state.state import AgentState
from src.tools.gnt_tools import INTERFERENCE_TOOLS


INTERFERENCE_SYSTEM = """You are the Interference Analyst Agent.

Your job is to identify the dominant aggressor when a cell shows interference-driven
degradation. You have one tool:
- compute_coupling: returns the directional ISR matrix for a victim cell

Workflow:
1. Call compute_coupling for the victim cell
2. Identify the dominant aggressor (highest incoming ISR)
3. Output a clear attribution: "The dominant aggressor for cell X is cell Y, ISR Z."

This is the structural insight that prevents compensatory loss in optimisation —
you're telling the operator to adjust the aggressor's parameters, not the symptom cell's.

When done, respond with INTERFERENCE FINDING: <attribution>
"""


def make_interference_analyst(llm: ChatAnthropic):
    llm_with_tools = llm.bind_tools(INTERFERENCE_TOOLS)

    def analyse(state: AgentState) -> dict:
        scenario = state["scenario"]
        diag = state.get("diagnostic_findings", {})

        prompt = f"""The diagnostician has identified a possible interference issue.

Scenario: {scenario.get('description')}
Affected cell: {scenario.get('affected_cell')}
Diagnostician's findings: {diag.get('diagnosis', 'none')}

Use compute_coupling to identify the dominant aggressor."""

        print(f"\n[INTERFERENCE ANALYST] Analysing aggressor for {scenario.get('affected_cell')}")

        messages = [
            SystemMessage(content=INTERFERENCE_SYSTEM),
            HumanMessage(content=prompt),
        ]

        findings = {}
        for _ in range(3):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                print(f"[INTERFERENCE ANALYST] Calling tool: {tc['name']}({tc['args']})")
                tool_fn = next((t for t in INTERFERENCE_TOOLS if t.name == tc["name"]), None)
                if tool_fn:
                    result = tool_fn.invoke(tc["args"])
                    findings[tc["name"]] = result
                    aggressor = result.get('dominant_aggressor', 'none')
                    print(f"[INTERFERENCE ANALYST]   -> dominant aggressor: {aggressor}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        analysis = response.content if response.content else "Analysis incomplete"
        print(f"[INTERFERENCE ANALYST] {analysis[:200]}")

        return {
            "messages": [AIMessage(content=f"INTERFERENCE: {analysis}")],
            "interference_findings": {
                "analysis": analysis,
                "tool_outputs": findings,
            },
            "next_agent": _next_agent(state),
        }

    return analyse


def _next_agent(state: AgentState) -> str:
    plan = state.get("routing_plan", [])
    if "interference_analyst" in plan:
        idx = plan.index("interference_analyst")
        if idx + 1 < len(plan):
            return plan[idx + 1]
    return "supervisor_compile"
