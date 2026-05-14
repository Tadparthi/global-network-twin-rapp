"""
Streamlit dashboard for the Global Network Twin multi-agent rApp API.

Run FastAPI first:
    python -m uvicorn src.api.main:app --reload

Then run this dashboard in a second terminal:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "claude-sonnet-4-5"


st.set_page_config(
    page_title="Global Network Twin rApp Dashboard",
    page_icon="📡",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        min-height: 98px;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.18rem;
        font-weight: 800;
        word-break: break-word;
    }
    .summary-box {
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 18px 20px;
        background: #f8fafc;
        color: #0f172a;
        line-height: 1.55;
        font-size: 1.02rem;
        font-weight: 500;
        word-break: normal;
        overflow-wrap: anywhere;
    }
    .route-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 12px;
        margin: 4px 6px 4px 0;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #3730a3;
        font-weight: 800;
        font-size: 0.88rem;
    }
    .arrow {
        color: #64748b;
        font-weight: 900;
        margin-right: 6px;
    }
    .policy-box {
        border: 1px solid #c7d2fe;
        border-radius: 16px;
        padding: 16px 18px;
        background: #eef2ff;
        color: #0f172a;
        line-height: 1.5;
        font-size: 0.98rem;
        font-weight: 500;
        overflow-wrap: anywhere;
    }
    .policy-title {
        color: #3730a3;
        font-size: 0.85rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .policy-json-note {
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=30)
def get_health(api_base_url: str) -> Dict[str, Any]:
    response = requests.get(f"{api_base_url}/health", timeout=8)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def get_scenarios(api_base_url: str) -> List[str]:
    response = requests.get(f"{api_base_url}/scenarios", timeout=8)
    response.raise_for_status()
    payload = response.json()
    return payload.get("scenarios", [])


def analyze_scenario(
    api_base_url: str,
    scenario_name: str,
    model: str,
    include_debug: bool,
    scenario_overrides: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "model": model,
        "include_debug": include_debug,
    }

    if scenario_overrides:
        payload["scenario_overrides"] = scenario_overrides

    response = requests.post(
        f"{api_base_url}/analyze-scenario",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def metric_card(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_get(mapping: Dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    current: Any = mapping or {}
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def extract_mock_a1_policy(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a dashboard-friendly policy view from the clean response and,
    when available, the optional debug/raw_state payload.
    """
    debug = result.get("debug") or {}
    raw_state = debug.get("raw_state") if isinstance(debug, dict) else {}

    # Some API versions may return raw_state directly inside debug, while others
    # may return the whole graph state. Support both shapes.
    state = raw_state if isinstance(raw_state, dict) else debug if isinstance(debug, dict) else {}

    policy_tool_output = _safe_get(
        state,
        "policy_output",
        "tool_outputs",
        "generate_a1_policy",
        default={},
    )

    risk = policy_tool_output.get("risk", {}) if isinstance(policy_tool_output, dict) else {}
    provenance = policy_tool_output.get("provenance", {}) if isinstance(policy_tool_output, dict) else {}
    cause = provenance.get("causeClassification", {}) if isinstance(provenance, dict) else {}

    policy = {
        "policy_id": policy_tool_output.get("policyId", "available in debug mode") if isinstance(policy_tool_output, dict) else "available in debug mode",
        "policy_class": policy_tool_output.get("policyClass", result.get("recommended_action", "review")) if isinstance(policy_tool_output, dict) else result.get("recommended_action", "review"),
        "target_cell": policy_tool_output.get("targetCell", result.get("affected_cell", "unknown")) if isinstance(policy_tool_output, dict) else result.get("affected_cell", "unknown"),
        "policy_status": result.get("policy_status", "unknown"),
        "requires_approval": result.get("human_review_required", "unknown"),
        "cause_class": cause.get("class", result.get("primary_issue", "unknown")),
        "confidence": cause.get("confidence", result.get("confidence", "unknown")),
        "temporal_urgency": risk.get("temporalUrgency", "available in debug mode") if isinstance(risk, dict) else "available in debug mode",
        "simulation_confidence": risk.get("simulationConfidence", "available in debug mode") if isinstance(risk, dict) else "available in debug mode",
        "capacity_headroom_pct": risk.get("capacityHeadroomPct", "available in debug mode") if isinstance(risk, dict) else "available in debug mode",
        "composite_risk_score": risk.get("compositeRiskScore", "available in debug mode") if isinstance(risk, dict) else "available in debug mode",
        "raw_policy": policy_tool_output if isinstance(policy_tool_output, dict) and policy_tool_output else None,
    }

    return policy


def render_mock_a1_policy_panel(result: Dict[str, Any], include_debug: bool) -> None:
    policy = extract_mock_a1_policy(result)

    st.subheader("Mock A1 Policy Output")
    st.caption(
        "This section separates O-RAN-style policy generation from the RF analysis summary. "
        "Enable debug output to view the full generated policy payload and provenance."
    )

    cols = st.columns(4)
    with cols[0]:
        metric_card("Policy ID", policy["policy_id"])
    with cols[1]:
        metric_card("Policy Class", policy["policy_class"])
    with cols[2]:
        metric_card("Target Cell", policy["target_cell"])
    with cols[3]:
        metric_card("Requires Approval", policy["requires_approval"])

    cols2 = st.columns(4)
    with cols2[0]:
        metric_card("Cause Class", policy["cause_class"])
    with cols2[1]:
        metric_card("Policy Status", policy["policy_status"])
    with cols2[2]:
        metric_card("Temporal Urgency", policy["temporal_urgency"])
    with cols2[3]:
        metric_card("Risk Score", policy["composite_risk_score"])

    st.markdown(
        f"""
        <div class="policy-box">
            <div class="policy-title">Policy intent</div>
            Recommended action: <strong>{result.get('recommended_action', 'review')}</strong><br/>
            Policy status: <strong>{result.get('policy_status', 'unknown')}</strong><br/>
            Human review required: <strong>{result.get('human_review_required', 'unknown')}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if policy.get("raw_policy"):
        with st.expander("Generated mock A1 policy JSON"):
            st.json(policy["raw_policy"], expanded=False)
    elif not include_debug:
        st.info("Enable 'Include debug output' in the sidebar to display full policyId, provenance, risk fields, and raw mock A1 JSON.")


def render_routing_plan(routing_plan: List[str]) -> None:
    if not routing_plan:
        st.info("No routing plan returned.")
        return

    html = ""
    for idx, step in enumerate(routing_plan):
        if idx > 0:
            html += '<span class="arrow">→</span>'
        html += f'<span class="route-pill">{step}</span>'

    st.markdown(html, unsafe_allow_html=True)


def parse_overrides(raw_text: str) -> Dict[str, Any] | None:
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON overrides: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Scenario overrides must be a JSON object.")
    return parsed


st.title("📡 Global Network Twin — Multi-Agent rApp Dashboard")
st.caption(
    "Portfolio demo UI for the FastAPI + LangGraph GNT rApp. "
    "The dashboard calls the API endpoint and displays a clean RF-planner summary."
)

with st.sidebar:
    st.header("API Settings")
    api_base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL)
    model = st.text_input("Anthropic model", value=DEFAULT_MODEL)

    st.divider()
    st.header("Scenario")

    scenarios: List[str] = []
    api_status = "Unknown"
    health_payload: Dict[str, Any] = {}

    try:
        health_payload = get_health(api_base_url)
        api_status = "Connected"
        scenarios = get_scenarios(api_base_url)
    except requests.exceptions.RequestException as exc:
        api_status = "Not connected"
        st.error(
            "Could not connect to the FastAPI service. Start it first with: "
            "python -m uvicorn src.api.main:app --reload"
        )
        st.caption(str(exc))

    st.write(f"API status: **{api_status}**")
    if health_payload:
        st.caption(f"Service: {health_payload.get('service', 'unknown')} | Version: {health_payload.get('version', 'unknown')}")

    scenario_name = st.selectbox(
        "Scenario",
        options=scenarios or ["congested_cell"],
        index=0,
    )

    include_debug = st.checkbox("Include debug output", value=False)

    with st.expander("Optional scenario overrides"):
        st.caption("Use JSON to override fields in the scenario file, for example demand_mbps or symptoms.")
        overrides_text = st.text_area(
            "Overrides JSON",
            value="",
            height=130,
            placeholder='{"demand_mbps": 3.0, "symptoms": "Moderate PRB pressure with stable SINR."}',
        )

    analyze_clicked = st.button("Run GNT Analysis", type="primary", use_container_width=True)


if not analyze_clicked:
    st.info("Choose a scenario in the sidebar and click **Run GNT Analysis**.")
    st.markdown(
        """
        **What this dashboard demonstrates:**

        1. FastAPI exposes the GNT rApp as a service.  
        2. LangGraph orchestrates supervisor and specialist agents.  
        3. GNT tools provide RF/domain logic.  
        4. The response is cleaned into a portfolio-ready RF-planner summary.  
        5. Optional debug mode shows raw agent/tool output.
        """
    )
    st.stop()


try:
    scenario_overrides = parse_overrides(overrides_text)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.spinner("Running LangGraph multi-agent GNT workflow..."):
    try:
        result = analyze_scenario(
            api_base_url=api_base_url,
            scenario_name=scenario_name,
            model=model,
            include_debug=include_debug,
            scenario_overrides=scenario_overrides,
        )
    except requests.exceptions.HTTPError as exc:
        st.error("API returned an error.")
        try:
            st.json(exc.response.json())
        except Exception:
            st.code(str(exc))
        st.stop()
    except requests.exceptions.RequestException as exc:
        st.error("Could not call the FastAPI endpoint.")
        st.code(str(exc))
        st.stop()

st.success("Analysis complete.")

row1 = st.columns(4)
with row1[0]:
    metric_card("Affected Cell", result.get("affected_cell", "unknown"))
with row1[1]:
    metric_card("Primary Issue", result.get("primary_issue", "unknown"))
with row1[2]:
    metric_card("Confidence", result.get("confidence", "unknown"))
with row1[3]:
    metric_card("Policy Status", result.get("policy_status", "unknown"))

row2 = st.columns(2)
with row2[0]:
    metric_card("Human Review Required", result.get("human_review_required", "unknown"))
with row2[1]:
    metric_card("Recommended Action", result.get("recommended_action", "review"))

st.subheader("Agent Routing Plan")
render_routing_plan(result.get("routing_plan", []))

st.subheader("RF Planner Summary")
st.markdown(
    f"<div class='summary-box'>{result.get('final_summary', 'No summary returned.')}</div>",
    unsafe_allow_html=True,
)

render_mock_a1_policy_panel(result, include_debug)

if include_debug:
    st.subheader("Debug Output")
    st.caption("Raw state, tool outputs, and agent findings returned by the API.")
    st.json(result.get("debug"), expanded=False)

with st.expander("Full clean API response"):
    st.json(result, expanded=False)
