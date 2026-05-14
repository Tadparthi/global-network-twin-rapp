# Architecture Notes

## Why supervisor pattern, not pipeline

A naive design would chain agents linearly: diagnostician → interference → capacity → policy. We don't do this because:

1. **Not every scenario needs every agent.** A pure capacity issue doesn't need the interference analyst. A pure mobility issue may not need the capacity planner.
2. **Routing is itself a decision.** Whether to escalate to interference analysis depends on what the diagnostician finds. The supervisor owns this routing logic.
3. **The supervisor compiles the final output.** Specialists produce structured findings; the supervisor produces planner-readable prose.

This is the standard pattern for production multi-agent systems and what hiring managers test for.

## State management

The shared `AgentState` (see `src/state/state.py`) is a TypedDict with explicit fields per agent:

```
AgentState
├── messages: list[BaseMessage]      # accumulating chat history (LangGraph reducer)
├── scenario: dict                    # input
├── routing_plan: list[str]           # supervisor's decision
├── diagnostic_findings: dict         # written by diagnostician
├── interference_findings: dict       # written by interference analyst
├── capacity_findings: dict           # written by capacity planner
├── policy_output: dict               # written by policy writer
├── final_recommendation: str         # written by supervisor at end
└── next_agent: str                   # routing pointer
```

Each agent writes only its own findings field. The supervisor reads all of them at the end. This avoids the typical multi-agent footgun where agents overwrite each other's state.

## Tool granularity

Each GNT module is one tool. Not "a giant analyse_cell tool that does everything" — each tool corresponds to one trained model with one well-defined input/output contract:

| Tool | Module | What it answers |
|---|---|---|
| `predict_sinr` | M1 | What's the SINR distribution for this cell? |
| `compute_coupling` | M2 | Which neighbour is the dominant aggressor? |
| `classify_degradation` | M3 | Coverage problem or capacity problem? |
| `classify_mobility` | M4 | Stationary, slow, or fast? |
| `compute_capacity` | M7 | PRB headroom and NPV-ranked interventions |
| `get_temporal_urgency` | M8 | Pattern → urgency level |
| `generate_a1_policy` | Step 8 | Package findings as A1 JSON |

Why this matters: the LLM can compose tools — call `compute_coupling` then `classify_degradation` then `compute_capacity` in sequence — without any tool needing to know about the others. This is the testable, debuggable pattern.

## Tool-use loop

Each agent runs a small tool-use loop (max 3-5 iterations):

```python
for _ in range(max_iterations):
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break  # Agent has settled on an answer

    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
```

This is the standard ReAct pattern — the agent calls tools, observes results, decides whether to call more tools or respond with a final answer. The bounded iteration count prevents runaway loops.

## Failure modes (what to watch for)

1. **Hallucinated tool results.** Mitigated by parsing tool outputs as structured data and printing them to console for verification.
2. **Infinite tool-call loops.** Bounded by `max_iterations` per agent (3-5).
3. **Routing errors.** The supervisor's JSON output is parsed defensively — if parsing fails, falls back to a default routing plan.
4. **Recursion limit.** LangGraph's `recursion_limit=25` prevents the graph itself from looping.

## Extending the system

To add a new specialist agent:

1. Create `src/agents/your_agent.py` following the pattern of the existing agents
2. Register the agent's tools in `src/tools/gnt_tools.py`
3. Add the node and routing in `src/graph.py`
4. Update the supervisor's system prompt to know about the new agent

To add a new tool:

1. Add the `@tool`-decorated function to `src/tools/gnt_tools.py`
2. Add it to the appropriate agent's tool list (e.g. `DIAGNOSTICIAN_TOOLS`)
3. Update the agent's system prompt to mention the new tool

## Production readiness

What this code IS:
- A working portfolio piece demonstrating supervisor + specialist multi-agent patterns
- A foundation you can extend with real GNT model outputs
- Concrete LangGraph + tool-use code suitable for technical interviews

What this code IS NOT:
- Production-deployable as-is (no auth, observability, retry logic, or cost controls)
- Connected to live ML models (uses mock data from your GNT JSON outputs)
- Tested with adversarial inputs

Adding production features (LangSmith observability, retry/timeout handling, auth, structured logging) is the next sprint.
