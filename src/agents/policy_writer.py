"""
Policy Writer Agent — packages findings into A1 policy.

Calls Module 8 (temporal urgency) and Step 8 (A1 generator) to produce
an O-RAN-compliant A1 policy with embedded provenance and risk score.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.state.state import AgentState
from src.tools.gnt_tools import POLICY_WRITER_TOOLS


POLICY_SYSTEM = """You are the Policy Writer Agent.

Your job is to package upstream findings into a properly-formed O-RAN A1 policy.
You have two tools:
- get_temporal_urgency: maps a temporal pattern to urgency level
- generate_a1_policy: produces the final A1 policy JSON with provenance

Workflow:
1. From the upstream findings, determine:
   - target_cell (the cell where the symptom appeared)
   - actual_aggressor (may differ from target if interference is the cause — from the
     interference analyst's findings)
   - cause (interference / geometry / mobility / capacity)
   - parameter_target and parameter_delta (what change to make)
2. Call get_temporal_urgency if a temporal pattern is mentioned
3. Call generate_a1_policy with all the parameters

Important: if the capacity verdict is CAPACITY_BOUND, the cause is "capacity" and the
policy is OUT_OF_BAND_REFARMING — flag requiresApproval.

When done, respond with POLICY GENERATED: <summary>
"""


def make_policy_writer(llm: ChatAnthropic):
    llm_with_tools = llm.bind_tools(POLICY_WRITER_TOOLS)

    def write(state: AgentState) -> dict:
        scenario = state["scenario"]
        diag = state.get("diagnostic_findings", {})
        interf = state.get("interference_findings", {})
        cap = state.get("capacity_findings", {})

        prompt = f"""Generate the A1 policy from these findings:

Scenario: {scenario.get('description')}
Target cell: {scenario.get('affected_cell')}

Diagnostician: {diag.get('diagnosis', 'none')}

Interference analyst: {interf.get('analysis', 'none')}

Capacity planner: {cap.get('verdict', 'none')}

Call get_temporal_urgency if relevant, then generate_a1_policy."""

        print(f"\n[POLICY WRITER] Packaging A1 policy for {scenario.get('affected_cell')}")

        messages = [
            SystemMessage(content=POLICY_SYSTEM),
            HumanMessage(content=prompt),
        ]

        findings = {}
        for _ in range(4):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                print(f"[POLICY WRITER] Calling tool: {tc['name']}({tc['args']})")
                tool_fn = next((t for t in POLICY_WRITER_TOOLS if t.name == tc["name"]), None)
                if tool_fn:
                    result = tool_fn.invoke(tc["args"])
                    findings[tc["name"]] = result
                    if "policyId" in result:
                        print(f"[POLICY WRITER]   -> A1 policy: {result['policyId']}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        summary = response.content if response.content else "Policy generation incomplete"
        print(f"[POLICY WRITER] {summary[:200]}")

        return {
            "messages": [AIMessage(content=f"POLICY: {summary}")],
            "policy_output": {
                "summary": summary,
                "tool_outputs": findings,
            },
            "next_agent": "supervisor_compile",
        }

    return write
