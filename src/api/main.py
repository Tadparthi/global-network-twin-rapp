"""
FastAPI deployment layer for the GNT multi-agent rApp.

Run locally:
    python -m uvicorn src.api.main:app --reload

Open docs:
    http://127.0.0.1:8000/docs
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # .env loading is optional. Environment variables may be set directly.
    pass

try:
    from anthropic import APIConnectionError, APIStatusError
except Exception:  # pragma: no cover - keeps API import-safe if anthropic changes exceptions
    APIConnectionError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore

from src.api.schemas import AnalyzeScenarioRequest, AnalyzeScenarioResponse
from src.graph import build_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"

app = FastAPI(
    title="Global Network Twin Multi-Agent rApp API",
    description=(
        "FastAPI service exposing the LangGraph-based Global Network Twin rApp: "
        "supervisor routing, RF specialist agents, GNT tool calls, policy generation, "
        "and final RF-planner recommendation."
    ),
    version="0.2.0",
)


def load_scenario(name: str) -> Dict[str, Any]:
    scenario_path = SCENARIOS_DIR / f"{name}.json"
    if not scenario_path.exists():
        available = sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Scenario not found: {name}",
                "available_scenarios": available,
            },
        )
    with open(scenario_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove LangChain message objects from API payloads.

    The full message history is useful internally, but it is noisy and not JSON-safe
    for a clean API response. The structured findings are what an external client
    should consume.
    """
    return {
        key: value
        for key, value in state.items()
        if key != "messages"
    }


def _confidence_label(value: Any) -> str:
    """Convert numeric confidence to a simple low/medium/high label."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _extract_primary_issue(final_state: Dict[str, Any]) -> str:
    """Extract a clean issue label from structured tool outputs when available."""
    capacity_tool = (
        final_state.get("capacity_findings", {})
        .get("tool_outputs", {})
        .get("classify_degradation", {})
    )
    verdict = capacity_tool.get("verdict")
    if verdict:
        return str(verdict).lower()

    policy_tool = (
        final_state.get("policy_output", {})
        .get("tool_outputs", {})
        .get("generate_a1_policy", {})
    )
    cause_class = (
        policy_tool.get("provenance", {})
        .get("causeClassification", {})
        .get("class")
    )
    if cause_class:
        return str(cause_class).lower()

    diagnosis = final_state.get("diagnostic_findings", {}).get("diagnosis", "")
    diagnosis_lower = diagnosis.lower()
    if "capacity" in diagnosis_lower:
        return "capacity_exhaustion"
    if "interference" in diagnosis_lower:
        return "interference_risk"
    if "mobility" in diagnosis_lower or "handover" in diagnosis_lower:
        return "mobility_risk"
    if "rf" in diagnosis_lower or "sinr" in diagnosis_lower:
        return "rf_degradation"
    return "unknown"


def _extract_confidence(final_state: Dict[str, Any]) -> str:
    capacity_tool = (
        final_state.get("capacity_findings", {})
        .get("tool_outputs", {})
        .get("classify_degradation", {})
    )
    if "confidence" in capacity_tool:
        return _confidence_label(capacity_tool.get("confidence"))

    policy_tool = (
        final_state.get("policy_output", {})
        .get("tool_outputs", {})
        .get("generate_a1_policy", {})
    )
    confidence = (
        policy_tool.get("provenance", {})
        .get("causeClassification", {})
        .get("confidence")
    )
    return _confidence_label(confidence)


def _extract_policy_status(final_state: Dict[str, Any]) -> str:
    policy_tool = (
        final_state.get("policy_output", {})
        .get("tool_outputs", {})
        .get("generate_a1_policy", {})
    )
    if policy_tool.get("requiresApproval") is True:
        return "requires_approval"

    summary = final_state.get("policy_output", {}).get("summary", "")
    summary_lower = summary.lower()
    if "requires approval" in summary_lower:
        return "requires_approval"
    if "blocked" in summary_lower:
        return "blocked"
    if "monitor" in summary_lower:
        return "monitor_only"
    if summary:
        return "generated"
    return "not_generated"


def _extract_human_review_required(final_state: Dict[str, Any]) -> bool:
    policy_tool = (
        final_state.get("policy_output", {})
        .get("tool_outputs", {})
        .get("generate_a1_policy", {})
    )
    if "requiresApproval" in policy_tool:
        return bool(policy_tool.get("requiresApproval"))
    return _extract_policy_status(final_state) in {"requires_approval", "blocked"}


def _strip_markdown(text: str) -> str:
    """Keep the API summary readable without returning full markdown sections."""
    if not text:
        return ""
    text = re.sub(r"[#*_`>-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_sentence_or_limit(text: str, max_chars: int = 650) -> str:
    clean = _strip_markdown(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _build_demo_summary(scenario: Dict[str, Any], final_state: Dict[str, Any]) -> str:
    """
    Build a concise, portfolio-friendly engineering summary from structured outputs.

    This intentionally avoids raw model metrics, finance-heavy language, truncated text,
    and unfinished LLM phrasing. The detailed agent/tool outputs remain available when
    include_debug=True.
    """

    affected_cell = scenario.get("affected_cell", "the affected cell")
    primary_issue = _extract_primary_issue(final_state)
    policy_status = _extract_policy_status(final_state)

    capacity_tool = (
        final_state.get("capacity_findings", {})
        .get("tool_outputs", {})
        .get("classify_degradation", {})
    )
    capacity_result = (
        final_state.get("capacity_findings", {})
        .get("tool_outputs", {})
        .get("compute_capacity", {})
    )
    sinr_result = (
        final_state.get("diagnostic_findings", {})
        .get("tool_outputs", {})
        .get("predict_sinr", {})
    )

    if primary_issue in {"capacity_bound", "capacity", "capacity_exhaustion", "capacity_pressure"}:
        current_ues = capacity_result.get("current_ue_count")
        demand_mbps = scenario.get("demand_mbps")
        mean_sinr = sinr_result.get("sinr_distribution", {}).get("mean_sinr_db")

        evidence_parts = []
        if current_ues is not None:
            evidence_parts.append(f"active UE load is high at {current_ues} UEs")
        if demand_mbps is not None:
            evidence_parts.append(f"observed demand is {demand_mbps} Mbps per UE")
        if mean_sinr is not None:
            evidence_parts.append(f"mean SINR is acceptable at {mean_sinr} dB")

        evidence = ", ".join(evidence_parts)
        if evidence:
            evidence = f" Key evidence: {evidence}."

        approval_note = (
            " Policy action requires human approval before any mock A1 output."
            if policy_status == "requires_approval"
            else " Policy output should remain under O-RAN guardrail review."
        )

        return (
            f"{affected_cell} shows capacity-bound degradation during the busy period. "
            f"RF quality appears acceptable, but traffic demand and PRB pressure exceed supportable capacity, "
            f"causing severe throughput collapse.{evidence} Recommend network planning review for capacity expansion, "
            f"carrier addition, refarming, or other capacity-relief strategy."
            f"{approval_note}"
        )

    if primary_issue in {"interference_risk", "interference"}:
        return (
            f"{affected_cell} shows an interference-driven degradation pattern. "
            f"The recommended next step is to review neighbor impact, interference coupling, SINR distribution, "
            f"and affected-cell dominance before generating any policy action. Policy output should remain under guardrail review."
        )

    if primary_issue in {"rf_degradation", "rf_quality_degradation"}:
        return (
            f"{affected_cell} shows an RF-quality degradation pattern. "
            f"The recommended next step is to review coverage, SINR/CQI/BLER behavior, and neighbor dominance before applying "
            f"capacity-oriented actions. Policy output should remain under guardrail review."
        )

    if primary_issue in {"mobility_risk", "mobility"}:
        return (
            f"{affected_cell} shows a mobility-risk pattern. "
            f"The recommended next step is to review handover behavior, serving-neighbor relationships, drop indicators, "
            f"and mobility guardrails before generating any policy action."
        )

    fallback = _first_sentence_or_limit(final_state.get("final_recommendation", ""), max_chars=500)
    return fallback or (
        f"{affected_cell} analysis completed. Review the debug output for detailed agent findings, tool outputs, "
        f"and policy-generation context."
    )


def _extract_recommended_action(final_state: Dict[str, Any]) -> str:
    capacity_tool = (
        final_state.get("capacity_findings", {})
        .get("tool_outputs", {})
        .get("classify_degradation", {})
    )
    action = capacity_tool.get("recommended_action_class")
    if action:
        return str(action)

    policy_tool = (
        final_state.get("policy_output", {})
        .get("tool_outputs", {})
        .get("generate_a1_policy", {})
    )
    policy_class = policy_tool.get("policyClass")
    if policy_class:
        return str(policy_class)

    return "review_recommendation"


def build_clean_response(
    request: AnalyzeScenarioRequest,
    scenario: Dict[str, Any],
    final_state: Dict[str, Any],
) -> AnalyzeScenarioResponse:
    """Build a concise response suitable for demos, README screenshots, and recruiters."""
    debug_payload = None
    if request.include_debug:
        debug_payload = {
            "diagnostic_findings": final_state.get("diagnostic_findings", {}),
            "interference_findings": final_state.get("interference_findings", {}),
            "capacity_findings": final_state.get("capacity_findings", {}),
            "policy_output": final_state.get("policy_output", {}),
            "raw_state": compact_state(final_state),
        }

    return AnalyzeScenarioResponse(
        scenario_name=request.scenario_name,
        affected_cell=scenario.get("affected_cell", "unknown"),
        routing_plan=final_state.get("routing_plan", []),
        primary_issue=_extract_primary_issue(final_state),
        confidence=_extract_confidence(final_state),
        policy_status=_extract_policy_status(final_state),
        human_review_required=_extract_human_review_required(final_state),
        final_summary=_build_demo_summary(scenario=scenario, final_state=final_state),
        recommended_action=_extract_recommended_action(final_state),
        debug=debug_payload,
    )


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "global-network-twin-multi-agent-rapp-api",
        "version": "0.2.0",
    }


@app.get("/scenarios")
def list_scenarios() -> Dict[str, List[str]]:
    scenarios = sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))
    return {"scenarios": scenarios}


@app.post("/analyze-scenario", response_model=AnalyzeScenarioResponse)
def analyze_scenario(request: AnalyzeScenarioRequest) -> AnalyzeScenarioResponse:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable is not set.",
        )

    scenario = load_scenario(request.scenario_name)
    if request.scenario_overrides:
        scenario.update(request.scenario_overrides)

    initial_state = {
        "messages": [],
        "scenario": scenario,
        "routing_plan": [],
        "diagnostic_findings": {},
        "interference_findings": {},
        "capacity_findings": {},
        "policy_output": {},
        "final_recommendation": "",
        "next_agent": "",
    }

    try:
        graph = build_graph(model_name=request.model, human_in_the_loop=False)
        final_state = graph.invoke(initial_state, config={"recursion_limit": 25})
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Anthropic API connection failed while running the LangGraph workflow. "
                "Check internet/VPN/firewall/API endpoint, then retry."
            ),
        ) from exc
    except APIStatusError as exc:
        status_code = getattr(exc, "status_code", 502) or 502
        raise HTTPException(
            status_code=status_code,
            detail=f"Anthropic API returned an error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"GNT workflow failed: {exc}",
        ) from exc

    return build_clean_response(request=request, scenario=scenario, final_state=final_state)
