# FastAPI Deployment Layer

This project exposes the LangGraph GNT multi-agent rApp through a FastAPI service.

## Run locally

```powershell
uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### `GET /health`

Checks that the API is running.

### `GET /scenarios`

Lists available scenario JSON files.

### `POST /analyze-scenario`

Runs the LangGraph multi-agent workflow for a scenario.

Example request:

```json
{
  "scenario_name": "congested_cell",
  "model": "claude-sonnet-4-5"
}
```

Example request with overrides:

```json
{
  "scenario_name": "congested_cell",
  "model": "claude-sonnet-4-5",
  "scenario_overrides": {
    "demand_mbps": 3.0,
    "symptoms": "Reduced throughput with moderate PRB pressure and stable SINR."
  }
}
```

## Docker

Build:

```bash
docker build -t gnt-rapp-api .
```

Run:

```bash
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="your_key_here" gnt-rapp-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Design

FastAPI exposes the service. LangGraph orchestrates the workflow. The existing GNT agents and tools remain the RF/KPI/policy logic.

## Clean vs Debug Response Mode

`POST /analyze-scenario` now returns a concise engineering summary by default. This is the preferred mode for demos, README screenshots, and resume/portfolio walkthroughs.

Default request:

```json
{
  "scenario_name": "congested_cell",
  "model": "claude-sonnet-4-5"
}
```

Default response includes:

- `scenario_name`
- `affected_cell`
- `routing_plan`
- `primary_issue`
- `confidence`
- `policy_status`
- `human_review_required`
- `recommended_action`
- `final_summary`

The clean response intentionally avoids raw model metrics, finance-heavy language, and truncated LLM text. It is designed for screenshots, demos, and GitHub/portfolio presentation.

To include the full raw LangGraph state, specialist findings, and tool outputs, set `include_debug` to `true`:

```json
{
  "scenario_name": "congested_cell",
  "model": "claude-sonnet-4-5",
  "include_debug": true
}
```

## Streamlit dashboard

The project also includes a lightweight Streamlit dashboard for portfolio/demo use.

Start the FastAPI service first:

```powershell
python -m uvicorn src.api.main:app --reload
```

Then open a second terminal and run:

```powershell
streamlit run src/dashboard/app.py
```

The dashboard calls:

```text
POST http://127.0.0.1:8000/analyze-scenario
```

Dashboard features:

- Scenario dropdown loaded from `/scenarios`
- Anthropic model field
- Debug mode toggle
- Optional scenario override JSON
- Agent routing plan display
- Primary issue, confidence, policy status, and human-review cards
- Clean RF-planner final summary
- Optional raw debug JSON view


## Streamlit Dashboard: Mock A1 Policy Panel

The Streamlit dashboard now separates the output into two visible sections:

1. **RF Planner Summary** — clean engineering analysis of the scenario.
2. **Mock A1 Policy Output** — O-RAN-style policy intent, policy status, target cell, approval requirement, and optional full policy JSON.

To view the full generated policy payload, enable **Include debug output** in the sidebar before running the analysis.
