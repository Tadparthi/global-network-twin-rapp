# GNT Multi-Agent rApp

> **Multi-agent LangGraph system for 5G RF optimisation.** Supervisor pattern with Claude Sonnet 4 coordinating four specialist agents that call physics-constrained ML models as tools to diagnose degraded cells and generate O-RAN A1 policies with embedded provenance.

![demo](demo.gif)

## What this is

A working multi-agent system demonstrating production-pattern agentic AI:

- **Supervisor pattern** — one coordinator agent decides which specialists to invoke based on the scenario, rather than running a fixed pipeline
- **Pydantic structured output** — the supervisor's routing decision is type-safe and validated against a schema
- **ReAct loops inside specialists** — each agent calls tools iteratively, deciding when it has enough information
- **Physics-grounded tools** — seven RF/network-engineering tools wrap trained ML models with well-defined input/output contracts
- **O-RAN A1 output** — generates policies with embedded provenance and risk scoring, ready for downstream xApp consumption
- **Human-in-the-loop approval** — optional `--approve-policy` mode pauses before policy generation via LangGraph's checkpointer + `interrupt_before` pattern, lets an operator review findings before A1 emission

The tools are wrappers around the Global Network Twin (GNT) — a physics-constrained 5G digital twin implementing TR 38.901 propagation with bounded ML residuals. Both this rApp and the deterministic pipeline implementation share the same A1 output contract; the agentic version adds adaptive routing.

## Architecture

```
                    ┌─────────────────────────┐
                    │   SUPERVISOR AGENT      │
                    │  (Pydantic structured   │
                    │     output routing)     │
                    └────────────┬────────────┘
                                 │ routes to:
            ┌────────────┬───────┴──────┬─────────────┐
            ▼            ▼              ▼             ▼
       ┌────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
       │ DIAG-  │   │ INTERF.  │   │ CAPACITY │   │ POLICY │
       │NOSTIC. │   │ ANALYST  │   │ PLANNER  │   │ WRITER │
       └───┬────┘   └────┬─────┘   └────┬─────┘   └────┬───┘
           │             │              │              │
           ▼             ▼              ▼              ▼
       ┌────────────────────────────────────────────────────┐
       │              GNT TOOLS (RF physics)                │
       │  predict_sinr · compute_coupling · classify_       │
       │  degradation · classify_mobility · compute_        │
       │  capacity · get_temporal_urgency · generate_a1     │
       └────────────────────────────────────────────────────┘
```

## Why the supervisor pattern

The naive multi-agent design chains agents in a linear pipeline. This system uses a coordinator-and-specialists pattern instead because:

1. **Not every scenario needs every agent.** A pure capacity issue doesn't need the interference analyst. A mobility issue might not need the capacity planner.
2. **Routing itself is a decision.** Whether to escalate to interference analysis depends on what the diagnostician finds. The supervisor owns this logic.
3. **The supervisor compiles the final output.** Specialists produce structured findings; the supervisor produces planner-readable prose.

This is the standard pattern for production multi-agent systems.

## What each agent does

| Agent | Role | Tools |
|---|---|---|
| **Supervisor** | Routes the scenario, compiles final recommendation | none (orchestrates only) |
| **Diagnostician** | Identifies what's wrong with a degraded cell | `classify_mobility`, `predict_sinr` |
| **Interference Analyst** | Identifies dominant aggressor cell | `compute_coupling` |
| **Capacity Planner** | Decides opex vs capex; computes NPV | `classify_degradation`, `compute_capacity` |
| **Policy Writer** | Packages findings into O-RAN A1 policy with provenance | `get_temporal_urgency`, `generate_a1_policy` |

## Quick start (Windows)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# If activation blocked:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Install dependencies
pip install -r requirements.txt

# Set API key
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Run a scenario (autonomous end-to-end)
python -m src.main --scenario congested_cell

# Run with human-in-the-loop approval before policy generation
python -m src.main --scenario congested_cell --approve-policy
```

Three scenarios available: `congested_cell`, `interference_spike`, `mobility_overload`.

The `--approve-policy` flag enables LangGraph's checkpointer + interrupt pattern. The graph runs the upstream agents (supervisor → diagnostician → capacity planner), pauses before the policy writer, prints the findings, and waits for operator approval (y/n) before generating the A1 policy. See `docs/checkpointing.md` for the pattern explanation and production upgrade path.

## Project layout

```
gnt-agent-rapp/
├── README.md                       This file
├── requirements.txt                Python dependencies
├── src/
│   ├── main.py                     Entry point with --approve-policy mode
│   ├── graph.py                    LangGraph state machine
│   ├── state/
│   │   └── state.py                Shared AgentState schema
│   ├── agents/
│   │   ├── supervisor.py           Coordinator (Pydantic structured output)
│   │   ├── diagnostician.py        Diagnoses degradation cause
│   │   ├── interference.py         Identifies aggressor cell
│   │   ├── capacity.py             Coverage vs capacity verdict
│   │   └── policy_writer.py        A1 policy generation
│   └── tools/
│       ├── gnt_tools.py            GNT modules as @tool functions
│       └── mock_data.py            Pre-computed outputs from GNT models
├── scenarios/
│   ├── congested_cell.json
│   ├── interference_spike.json
│   └── mobility_overload.json
└── docs/
    ├── architecture.md             Design notes
    ├── structured_output.md        Why Pydantic over regex JSON
    ├── checkpointing.md            Human-in-the-loop pattern
    └── quickstart.md               Windows cheat sheet
```

## Production considerations

What this code IS:
- A working demonstration of supervisor + specialist multi-agent patterns
- A foundation that swaps mock tool outputs for live model calls without architectural changes
- Production-pattern code (typed state, structured output, bounded ReAct loops, conditional routing, checkpointer + interrupt for HITL)

What this code IS NOT:
- Production-deployed (no auth, observability, retry/timeout handling)
- Connected to live ML models (tools return outputs sourced from the GNT models' actual JSON results)
- Tested with adversarial inputs
- Durably persisted (uses `MemorySaver` for the portfolio piece — swap for `SqliteSaver`/`PostgresSaver` in production)

To deploy in production: add LangSmith for observability, replace `MemorySaver` with `SqliteSaver` or `PostgresSaver` for state persistence across restarts, swap mock tools for live model calls, containerise.

## Tech stack

- **Python 3.11+**
- **LangGraph** — state machine for multi-agent orchestration, checkpointer + interrupt
- **LangChain** — tool wrappers, message types, structured output
- **Anthropic Claude Sonnet 4** — LLM provider
- **Pydantic** — schemas for type-safe routing decisions

## Domain context

This system implements the agentic version of an O-RAN Non-RT RIC rApp. The rApp consumes O1 data from the SMO and emits A1 policies to downstream xApps (Coverage Optimisation, Interference Coordination, Mobility Management). Same output contract as a deterministic rApp implementation, but with adaptive routing — the supervisor decides which specialists to invoke based on the scenario rather than running a fixed pipeline.

If you're not from telecom: the same pattern works for customer support automation, financial research, code review, legal contract review, industrial IoT — anywhere you need adaptive routing through specialist agents that call domain-specific tools.

## License

MIT

## FastAPI deployment layer

This repository also includes a thin FastAPI wrapper around the LangGraph workflow.

Run locally:

```powershell
uvicorn src.api.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Main endpoint:

```text
POST /analyze-scenario
```

The API executes the same multi-agent GNT workflow: supervisor routing, specialist tool calls, policy generation, and final RF-planner recommendation.

Docker:

```bash
docker build -t gnt-rapp-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="your_key_here" gnt-rapp-api
```

See `docs/api_deployment.md` for details.

## API Response Modes

The FastAPI service supports two response modes for `POST /analyze-scenario`:

- **Clean mode**: default, concise engineering summary suitable for demos and portfolio screenshots.
- **Debug mode**: enabled with `include_debug: true`, returns raw LangGraph state, specialist findings, and detailed tool outputs.

Example clean request:

```json
{
  "scenario_name": "congested_cell",
  "model": "claude-sonnet-4-5"
}
```

Example debug request:

```json
{
  "scenario_name": "congested_cell",
  "model": "claude-sonnet-4-5",
  "include_debug": true
}
```

## Streamlit dashboard

A lightweight Streamlit dashboard is included for portfolio/demo screenshots.

Run the API first:

```powershell
python -m uvicorn src.api.main:app --reload
```

Then run the dashboard in a second terminal:

```powershell
streamlit run src/dashboard/app.py
```

The dashboard provides scenario selection, model selection, supervisor-agent routing visualization, RF diagnosis cards, O-RAN policy approval status, final RF-planner summary, and optional debug output.


### Streamlit Dashboard — Mock A1 Policy Panel

The dashboard includes a dedicated **Mock A1 Policy Output** section so the demo clearly shows both the RF analysis result and the O-RAN-style policy generation result. The panel shows policy class, target cell, approval requirement, cause class, policy status, temporal urgency, and risk score where available.

Enable **Include debug output** in the dashboard sidebar to display the full generated mock A1 policy JSON and provenance fields.
