"""
Pydantic request/response schemas for the GNT multi-agent rApp API.

Default API behavior is demo-friendly: return a concise engineering summary.
Set include_debug=True to include the full raw LangGraph state/tool outputs.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalyzeScenarioRequest(BaseModel):
    scenario_name: str = Field(
        default="congested_cell",
        description="Scenario JSON name from the scenarios directory, without .json.",
    )
    model: str = Field(
        default="claude-sonnet-4-5",
        description="Anthropic model identifier used by the supervisor and specialist agents.",
    )
    scenario_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional scenario fields to override for testing, such as demand_mbps or symptoms.",
    )
    include_debug: bool = Field(
        default=False,
        description=(
            "When false, return a concise portfolio/demo-ready response. "
            "When true, include raw LangGraph state and detailed agent/tool outputs."
        ),
    )


class AnalyzeScenarioResponse(BaseModel):
    scenario_name: str
    affected_cell: str
    routing_plan: List[str]

    primary_issue: str
    confidence: str
    policy_status: str
    human_review_required: bool

    final_summary: str
    recommended_action: str

    debug: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full raw state/tool outputs. Present only when include_debug=True.",
    )
