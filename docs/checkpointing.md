# Checkpointing and Human-in-the-Loop

This project supports human-in-the-loop approval via LangGraph's checkpointer
and interrupt pattern. Worth understanding because it's the feature that
differentiates "I built a demo agent" from "I've thought about deploying agents
in production".

## When to use it

Pass `--approve-policy` to enable human-in-the-loop mode:

```powershell
python -m src.main --scenario congested_cell --approve-policy
```

The graph runs the upstream agents (supervisor → diagnostician → capacity
planner → ...) and pauses **before** the policy_writer node. The console
prints the upstream findings and waits for operator approval (y/n) before
generating the A1 policy.

This maps onto the original GNT spec's Daily Brief approval step. The RF
planner reviews the diagnostic findings, validates them against operational
context the agents don't have visibility into, and approves only the policies
that should go to xApps.

## How it works

Two LangGraph features wired together:

### 1. Checkpointer

A checkpointer persists state after every node executes. This lets the graph
pause and resume across multiple invocations — without it, the graph either
runs to completion in one call or fails entirely.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

Each conversation (or scenario run) is identified by a `thread_id`. The
checkpointer saves state under that ID; resuming with the same ID restores
the saved state.

### 2. Interrupt

Marks specific nodes as pause points. Execution stops just before the named
node and returns to the caller.

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["policy_writer"],
)
```

In our system, the interrupt is configured to fire just before the
`policy_writer` node. The agents upstream complete their analysis; the
graph pauses with the findings in state; the caller decides whether to
resume or halt.

## The execution flow

```
First invocation:
graph.invoke(initial_state, config={"configurable": {"thread_id": "scenario-001"}})

  supervisor_route → diagnostician → capacity_planner → [INTERRUPT]
                                                            ↓
                                            state saved by checkpointer
                                            control returned to caller

Caller reviews findings, prompts operator for y/n

If approved:
graph.invoke(None, config={"configurable": {"thread_id": "scenario-001"}})

  [resume from checkpoint] → policy_writer → supervisor_compile → END
```

The `None` input on the second invocation is the signal to resume from the
last checkpoint. The thread_id must match the first call so the checkpointer
knows which state to restore.

## Production considerations

What we use vs what you'd use in production:

| Aspect | Portfolio (this project) | Production |
|---|---|---|
| Checkpointer | `MemorySaver` (in-process) | `SqliteSaver` or `PostgresSaver` (durable) |
| Approval UI | CLI prompt (`input()`) | Web UI, Slack approval, ticket workflow |
| Thread management | UUID per scenario | User-scoped threads with auth |
| Audit trail | Console output | Structured logs + LangSmith traces |
| Timeout handling | None (blocks indefinitely) | Approval timeout with default action |

The `MemorySaver` is fine for demonstrating the pattern. It loses state if
the Python process exits — a real operator workflow needs `SqliteSaver` or
`PostgresSaver` so paused scenarios survive deployment restarts.

To upgrade: replace these two lines in `src/graph.py`:

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```

with:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("/path/to/checkpoints.db")
```

The rest of the graph is unchanged — that's the value of LangGraph's
abstraction.

## Why this matters in interviews

When an interviewer asks "how would you handle approval workflows in
production?", the answer that lands:

> "LangGraph's checkpointer + interrupt pattern. Compile the graph with a
> persistent checkpointer (SQLite or Postgres), mark nodes that need
> approval with `interrupt_before`, identify each scenario with a thread_id.
> The graph runs to the interrupt and pauses; the approval system (UI, Slack,
> ticket) retrieves the findings; on approval the graph resumes with the
> same thread_id. On rejection the scenario is closed without policy
> generation. State survives process restarts because the checkpointer
> persists everything."

That answer demonstrates you've thought about real deployment: persistence,
scenario identity, the approval UI as a separate concern from the graph,
and graceful handling of rejections. Most candidates haven't gotten that far.

## What's still missing

This implementation demonstrates the pattern but doesn't include:

- **State editing during approval** — the operator can approve or reject but
  can't modify findings before resumption. Production would allow editing.
- **Multi-step approvals** — currently only `policy_writer` is interrupted.
  A real workflow might have approvals at multiple stages (e.g., before any
  capex-class decision).
- **Approval timeouts** — the CLI blocks indefinitely waiting for input.
  Production needs a timeout with a default action (auto-approve low-risk,
  auto-reject high-risk).
- **Audit trail** — who approved, when, with what context. Production logs
  this for regulatory compliance.

These are the obvious next steps if this were going to a real operator.
