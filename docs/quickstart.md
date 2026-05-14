# Quick Start (Windows)

## First-time setup

```powershell
# 1. Create venv
python -m venv venv

# 2. Activate
.\venv\Scripts\Activate.ps1

# 3. If activation blocked:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 4. Install deps
pip install -r requirements.txt

# 5. Set API key
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Run a scenario

```powershell
python -m src.main --scenario congested_cell
python -m src.main --scenario interference_spike
python -m src.main --scenario mobility_overload
```

## Common errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: langgraph` | `pip install -r requirements.txt` |
| `AuthenticationError` | API key not set or invalid |
| `pip not found` | Reinstall Python with "Add to PATH" checked |
| `cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ImportError: cannot import name 'tool'` | LangChain version mismatch — `pip install --upgrade langchain langchain-anthropic` |

## What good output looks like

You should see something like:

```
======================================================================
  RF MULTI-AGENT SYSTEM — Scenario: congested_cell
======================================================================
  Model: claude-sonnet-4-5
  Description: Cell A is showing severely degraded UE throughput...
======================================================================

[SUPERVISOR] Scenario received: congested_cell
[SUPERVISOR] Routing plan: diagnostician -> capacity_planner -> policy_writer

[DIAGNOSTICIAN] Starting analysis for cell_A
[DIAGNOSTICIAN] Calling tool: predict_sinr({'cell_id': 'cell_A'})
[DIAGNOSTICIAN]   -> {'cell_id': 'cell_A', 'sinr_distribution': {'mean_sinr_db': 5.7,...

[CAPACITY PLANNER] Computing verdict for cell_A
[CAPACITY PLANNER] Calling tool: classify_degradation({'cell_id': 'cell_A',...
[CAPACITY PLANNER]   -> CAPACITY_BOUND (0.94)
[CAPACITY PLANNER] Calling tool: compute_capacity({'cell_id': 'cell_A'})
[CAPACITY PLANNER]   -> 244 UEs vs 5 supportable — 48x oversubscribed

[POLICY WRITER] Packaging A1 policy for cell_A
[POLICY WRITER] Calling tool: generate_a1_policy(...)
[POLICY WRITER]   -> A1 policy: SAND-GNT-CAPACITY-A-001

[SUPERVISOR] Final recommendation compiled.

======================================================================
  FINAL RECOMMENDATION
======================================================================
[Claude's compiled recommendation appears here]
```

The agents stream their reasoning so you can see the tool calls happening in real time. This is what makes it interview-demonstrable.

## Next steps after it runs

1. Push the project to a public GitHub repo
2. Record a short screen-cap of the agents running (loom, OBS, or PowerShell screen-recorder)
3. Add `demo.gif` to the repo
4. Write a LinkedIn post linking to the repo
5. Add the project to your CV with a short description
6. Mention it in agent-AI / multi-agent role applications

The portfolio claim becomes:

> Built a supervisor-pattern multi-agent system in LangGraph using Claude Sonnet 4. Four specialist agents call physics-constrained ML tools to diagnose 5G RF issues and generate O-RAN A1 policies with embedded provenance.
