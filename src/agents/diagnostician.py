"""
Diagnostician Agent — identifies what's wrong.

Calls Module 4 (mobility classifier) and Module 1 (SINR predictor) as tools
to determine whether the degradation is caused by mobility, geometry, or
something deeper that needs the interference analyst.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state.state import AgentState
from src.tools.gnt_tools import DIAGNOSTICIAN_TOOLS


DIAGNOSTICIAN_SYSTEM = """You are the Diagnostician Agent.

Your job is to identify what's wrong with a degraded cell. You have two tools:
- classify_mobility: classifies UE trajectory state from speed, displacement, handovers
- predict_sinr: returns the SINR distribution for a cell from M1's physics-constrained model

Workflow:
1. Call predict_sinr for the affected cell to see the SINR distribution
2. If the scenario mentions UE movement or handover failures, call classify_mobility
3. Synthesise: is this a mobility issue, a geometry issue, or does it need deeper analysis?

Output a concise diagnosis with the cause hypothesis. Be specific — cite actual SINR
numbers and confidence levels from the tool calls. Don't speculate beyond what the
tools tell you.

When you've completed your analysis, respond with FINAL DIAGNOSIS: <your diagnosis>
"""


def make_diagnostician(llm: ChatAnthropic):
    """Returns the diagnostician node function with tools bound."""
    llm_with_tools = llm.bind_tools(DIAGNOSTICIAN_TOOLS)

    def diagnose(state: AgentState) -> dict:
        scenario = state["scenario"]
        prompt = f"""Diagnose this scenario:

{scenario.get('description')}

Affected cell: {scenario.get('affected_cell')}
Symptoms: {scenario.get('symptoms')}
Trajectory data (if relevant): {scenario.get('trajectory_data', 'not provided')}

Use your tools to gather evidence, then provide your final diagnosis."""

        print(f"\n[DIAGNOSTICIAN] Starting analysis for {scenario.get('affected_cell')}")

        messages = [
            SystemMessage(content=DIAGNOSTICIAN_SYSTEM),
            HumanMessage(content=prompt),
        ]

        # Tool-use loop — let the agent call multiple tools before settling
        max_iterations = 5
        findings = {}

        for i in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # Agent has finished
                break

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"[DIAGNOSTICIAN] Calling tool: {tool_name}({tool_args})")

                # Find and run the tool
                tool_fn = next((t for t in DIAGNOSTICIAN_TOOLS if t.name == tool_name), None)
                if tool_fn:
                    result = tool_fn.invoke(tool_args)
                    findings[tool_name] = result
                    print(f"[DIAGNOSTICIAN]   -> {_short(result)}")

                    # Append tool result for the next LLM iteration
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    ))

        diagnosis = response.content if response.content else "Diagnosis incomplete"
        print(f"[DIAGNOSTICIAN] {_short(diagnosis, 200)}")

        return {
            "messages": [AIMessage(content=f"DIAGNOSTIC: {diagnosis}")],
            "diagnostic_findings": {
                "diagnosis": diagnosis,
                "tool_outputs": findings,
            },
            "next_agent": _determine_next(state, findings),
        }

    return diagnose


def _short(s, max_len=120):
    """Truncate strings/dicts for console output."""
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + "..."


def _determine_next(state: AgentState, findings: dict) -> str:
    """Decide which agent runs next based on the routing plan."""
    plan = state.get("routing_plan", [])
    if "diagnostician" in plan:
        idx = plan.index("diagnostician")
        if idx + 1 < len(plan):
            return plan[idx + 1]
    return "supervisor_compile"
