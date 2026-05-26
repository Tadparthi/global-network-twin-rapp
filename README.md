# GNT Multi-Agent rApp

![CI](https://github.com/Tadparthi/global-network-twin-rapp/actions/workflows/ci.yml/badge.svg)

> **Multi-agent LangGraph system for 5G RF optimisation.** Supervisor pattern with Claude Sonnet 4 coordinating four specialist agents that call physics-constrained ML models as tools to diagnose degraded cells and generate O-RAN A1 policies with embedded provenance.


## What this is

A working multi-agent system demonstrating production-pattern agentic AI:

- **Supervisor pattern** — one coordinator agent decides which specialists to invoke based on the scenario, rather than running a fixed pipeline
- **Pydantic structured output** — the supervisor's routing decision is type-safe and validated against a schema
- **ReAct loops inside specialists** — each agent calls tools iteratively, deciding when it has enough information
- **Physics-grounded tools** — seven RF/network-engineering tools wrap trained ML models with well-defined input/output contracts
- **MCP server** — exposes the GNT backend as standardised tools for any MCP-compatible client
- **O-RAN A1 output** — generates policies with embedded provenance and risk scoring, ready for downstream xApp consumption
- **Human-in-the-loop approval** — optional `--approve-policy` mode pauses before policy generation via LangGraph's checkpointer + `interrupt_before` pattern, lets an operator review findings before A1 emission
- **Tested and CI-backed** — unit test suite covering the tool layer and API endpoints, run automatically on every push via GitHub Actions

The tools are wrappers around the Global Network Twin (GNT) — a physics-constrained 5G digital twin implementing TR 38.901 propagation with bounded ML residuals. Both this rApp and the deterministic pipeline implementation share the same A1 output contract; the agentic version adds adaptive routing.

## Architecture

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
  
   Two distinct layers do two distinct jobs. **LangGraph** orchestrates *which* agent runs and when — the supervisor's conditional routing. **MCP** is the *transport* through which an external client reaches the GNT backend. They are not the same mechanism described twice: LangGraph is internal orchestration, MCP is the external tool interface.

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

## MCP server

The repository includes a [Model Context Protocol](https://modelcontextprotocol.io) server (`src/mcp_server/server.py`, built with `fastmcp`) that exposes the deployed GNT backend to any MCP-compatible client — Claude Desktop, IDEs, or other agents.

The server is a thin, stateless HTTP client: it does not import GNT code directly, but calls the deployed FastAPI service over REST. This keeps the MCP layer fully decoupled from the backend. It exposes three high-level tools:

| MCP tool | Backend endpoint | Purpose |
|---|---|---|
| `health_check_gnt_api` | `GET /health` | Check the GNT backend is reachable |
| `list_gnt_scenarios` | `GET /scenarios` | List available analysis scenarios |
| `analyze_gnt_scenario` | `POST /analyze-scenario` | Run the full multi-agent analysis for a scenario |

Rather than surfacing every low-level GNT module individually, the server exposes coarse-grained, task-level operations — keeping the deterministic RF engine encapsulated behind a clean interface.

The backend URL defaults to the deployed instance and is overridable via the `GNT_API_BASE_URL` environment variable:

```powershell
# Point the MCP server at a local backend instead of the deployed one
$env:GNT_API_BASE_URL = "http://127.0.0.1:8000"
python -m src.mcp_server.server
```
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

# Run the test suite
pytest -v
```

Three scenarios available: `congested_cell`, `interference_spike`, `mobility_overload`.

The `--approve-policy` flag enables LangGraph's checkpointer + interrupt pattern. The graph runs the upstream agents (supervisor → diagnostician → capacity planner), pauses before the policy writer, prints the findings, and waits for operator approval (y/n) before generating the A1 policy. See `docs/checkpointing.md` for the pattern explanation and production upgrade path.

## Testing

The project has a unit test suite covering the deterministic layers:

- **`tests/test_gnt_tools.py`** — verifies the seven GNT tools: SINR prediction bounds, interference-aggressor identification, coverage/capacity → opex/capex mapping, mobility thresholds, A1 policy structure and the capacity-cause approval guardrail.
- **`tests/test_api.py`** — verifies the FastAPI endpoints, including clean error handling when the API key is absent or a scenario name is unknown.

The suite deliberately does not call the live LLM — non-deterministic model output cannot be asserted on — so it covers the deterministic tools, routing wiring, schemas, and error paths. Tests run automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

```powershell
pytest -v
```

## Project layout
global-network-twin-rapp/
├── README.md                       This file
├── requirements.txt                Python dependencies
├── Dockerfile                      Container build for the API
├── .github/
│   └── workflows/
│       └── ci.yml                  GitHub Actions CI — runs the test suite
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
│   ├── tools/
│   │   ├── gnt_tools.py            GNT modules as @tool functions
│   │   └── mock_data.py            Pre-computed outputs from GNT models
│   ├── mcp_server/                 MCP server exposing the GNT backend
│   ├── api/
│   │   └── main.py                 FastAPI wrapper
│   └── dashboard/
│       └── app.py                  Streamlit demo dashboard
├── tests/
│   ├── test_gnt_tools.py           Unit tests for the GNT tool layer
│   └── test_api.py                 API endpoint tests
├── scenarios/
│   ├── congested_cell.json
│   ├── interference_spike.json
│   └── mobility_overload.json
└── docs/
├── architecture.md             Design notes
├── structured_output.md        Why Pydantic over regex JSON
├── checkpointing.md            Human-in-the-loop pattern
└── quickstart.md               Windows cheat sheet
## Production considerations

What this code IS:
- A working demonstration of supervisor + specialist multi-agent patterns
- A foundation that swaps mock tool outputs for live model calls without architectural changes
- Production-pattern code (typed state, structured output, bounded ReAct loops, conditional routing, checkpointer + interrupt for HITL)
- Unit-tested at the tool and API layers, with CI on every push and structured logging across the agents

What this code IS NOT:
- Production-deployed (no auth, no metrics/tracing instrumentation, no retry/timeout handling)
- Connected to live ML models (tools return outputs sourced from the GNT models' actual JSON results)
- Fuzz-tested or adversarially hardened (the test suite covers expected and error-path behaviour, not adversarial inputs)
- Durably persisted (uses `MemorySaver` for the portfolio piece — swap for `SqliteSaver`/`PostgresSaver` in production)

To deploy in production: add LangSmith or OpenTelemetry for observability, replace `MemorySaver` with `SqliteSaver` or `PostgresSaver` for state persistence across restarts, swap mock tools for live model calls, add auth and retry/timeout handling, containerise.

## Tech stack

- **Python 3.11+** (CI runs 3.13)
- **LangGraph** — state machine for multi-agent orchestration, checkpointer + interrupt
- **LangChain** — tool wrappers, message types, structured output
- **MCP (Model Context Protocol)** — `fastmcp` server exposing the GNT backend as standardised tools
- **Anthropic Claude Sonnet 4** — LLM provider
- **Pydantic** — schemas for type-safe routing decisions
- **FastAPI** — HTTP wrapper around the workflow
- **Streamlit** — demo dashboard
- **pytest** — test suite, run via GitHub Actions

## FastAPI deployment layer

This repository includes a thin FastAPI wrapper around the LangGraph workflow.

Run locally:

```powershell
uvicorn src.api.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Main endpoint: `POST /analyze-scenario`. The API executes the same multi-agent GNT workflow — supervisor routing, specialist tool calls, policy generation, and final RF-planner recommendation.

Docker:

```bash
docker build -t gnt-rapp-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="your_key_here" gnt-rapp-api
```

See `docs/api_deployment.md` for details.

### API response modes

`POST /analyze-scenario` supports two response modes:

- **Clean mode** (default) — concise engineering summary suitable for demos and screenshots.
- **Debug mode** (`include_debug: true`) — returns raw LangGraph state, specialist findings, and detailed tool outputs.

```json
{ "scenario_name": "congested_cell", "model": "claude-sonnet-4-5" }
```

```json
{ "scenario_name": "congested_cell", "model": "claude-sonnet-4-5", "include_debug": true }
```

## Streamlit dashboard

A lightweight Streamlit dashboard is included for demo screenshots. Run the API first, then the dashboard in a second terminal:

```powershell
python -m uvicorn src.api.main:app --reload
streamlit run src/dashboard/app.py
```

The dashboard provides scenario selection, model selection, supervisor routing visualisation, RF diagnosis cards, O-RAN policy approval status, the final RF-planner summary, and optional debug output.

### Mock A1 policy panel

The dashboard includes a dedicated **Mock A1 Policy Output** section so the demo shows both the RF analysis result and the O-RAN-style policy generation result — policy class, target cell, approval requirement, cause class, policy status, temporal urgency, and risk score. Enable **Include debug output** in the sidebar to display the full generated mock A1 policy JSON and provenance fields.

## Domain context

This system implements the agentic version of an O-RAN Non-RT RIC rApp. The rApp consumes O1 data from the SMO and emits A1 policies to downstream xApps (Coverage Optimisation, Interference Coordination, Mobility Management). Same output contract as a deterministic rApp implementation, but with adaptive routing — the supervisor decides which specialists to invoke based on the scenario rather than running a fixed pipeline.

If you're not from telecom: the same pattern works for customer support automation, financial research, code review, legal contract review, industrial IoT — anywhere you need adaptive routing through specialist agents that call domain-specific tools.

## License

MIT