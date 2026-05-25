"""
Unit tests for the GNT tools layer (src/tools/gnt_tools.py).

These cover the deterministic, physics-grounded tools the agents call.
No LLM calls and no mocking are needed — every tool is a pure function
over the pre-computed GNT module outputs in mock_data.py.

Run:  pytest tests/test_gnt_tools.py -v
"""
import pytest

from src.tools.gnt_tools import (
    predict_sinr,
    compute_coupling,
    classify_degradation,
    classify_mobility,
    compute_capacity,
    get_temporal_urgency,
    generate_a1_policy,
    DIAGNOSTICIAN_TOOLS,
    INTERFERENCE_TOOLS,
    CAPACITY_TOOLS,
    POLICY_WRITER_TOOLS,
    ALL_TOOLS,
)

# LangChain @tool-decorated functions are invoked via .invoke(dict),
# not called directly. This helper keeps the tests readable.
def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ─────────────────────────────────────────────────────────────
# Module 1 — SINR Engine
# ─────────────────────────────────────────────────────────────

def test_predict_sinr_returns_known_baseline_for_cell_a():
    result = call(predict_sinr, cell_id="cell_A")
    assert result["cell_id"] == "cell_A"
    # mean SINR for cell_A is 5.7 dB in the GNT baseline
    assert result["sinr_distribution"]["mean_sinr_db"] == 5.7
    assert result["sinr_distribution"]["n_ues"] == 244

def test_predict_sinr_includes_ood_rmse_bound():
    # The 2.09 dB OOD bound is the headline physics-validation number —
    # this test pins it so a regression in mock_data is caught.
    result = call(predict_sinr, cell_id="cell_B")
    assert result["rmse_bounds"]["out_of_distribution"] == 2.09
    assert result["rmse_bounds"]["in_distribution"] < result["rmse_bounds"]["out_of_distribution"]


# ─────────────────────────────────────────────────────────────
# Module 2 — Interference Coupling
# ─────────────────────────────────────────────────────────────

def test_compute_coupling_identifies_dominant_aggressor():
    # For victim cell_A, cell_C->cell_A (ISR 0.223) outweighs cell_B->cell_A (0.137).
    result = call(compute_coupling, victim_cell="cell_A")
    assert result["dominant_aggressor"] == "cell_C"

def test_compute_coupling_aggressor_is_never_the_victim():
    # A cell cannot be its own dominant interferer — guards the directional logic.
    for cell in ("cell_A", "cell_B", "cell_C"):
        result = call(compute_coupling, victim_cell=cell)
        assert result["dominant_aggressor"] != cell

def test_compute_coupling_interpretation_names_the_aggressor():
    result = call(compute_coupling, victim_cell="cell_A")
    assert "cell_C" in result["interpretation"]


# ─────────────────────────────────────────────────────────────
# Module 3 — Coverage / Capacity Classifier
# ─────────────────────────────────────────────────────────────

def test_classify_degradation_capacity_bound_maps_to_capex():
    # cell_A at 8 Mbps is CAPACITY_BOUND -> must imply CAPEX.
    result = call(classify_degradation, cell_id="cell_A", demand_mbps=8.0)
    assert result["verdict"] == "CAPACITY_BOUND"
    assert "CAPEX" in result["implication"]

def test_classify_degradation_coverage_bound_maps_to_opex():
    # cell_A at 3 Mbps is COVERAGE_BOUND -> must imply OPEX.
    result = call(classify_degradation, cell_id="cell_A", demand_mbps=3.0)
    assert result["verdict"] == "COVERAGE_BOUND"
    assert "OPEX" in result["implication"]

def test_classify_degradation_unknown_key_falls_back_safely():
    # An unmapped demand level must not raise — it falls back to a default verdict.
    result = call(classify_degradation, cell_id="cell_B", demand_mbps=99.0)
    assert result["verdict"] in ("CAPACITY_BOUND", "COVERAGE_BOUND")


# ─────────────────────────────────────────────────────────────
# Module 4 — Mobility Classifier (threshold logic)
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("speed,expected", [
    (0.3, "stationary"),    # below 0.8 m/s
    (1.5, "slow_moving"),   # between 0.8 and 2.5
    (4.2, "fast_moving"),   # above 2.5 (the mobility_overload scenario speed)
])
def test_classify_mobility_threshold_boundaries(speed, expected):
    result = call(classify_mobility, speed_mps=speed,
                  displacement_m=100.0, n_handovers=2)
    assert result["mobility_class"] == expected

def test_classify_mobility_fast_routes_to_mobility_xapp():
    result = call(classify_mobility, speed_mps=5.0,
                  displacement_m=400.0, n_handovers=5)
    assert "Mobility Management" in result["routing"]


# ─────────────────────────────────────────────────────────────
# Module 7 — Capacity Oracle
# ─────────────────────────────────────────────────────────────

def test_compute_capacity_reports_oversubscription_for_cell_a():
    # cell_A: 244 active UEs vs 5 supportable at heavy demand — heavily oversubscribed.
    result = call(compute_capacity, cell_id="cell_A")
    assert result["current_ue_count"] == 244
    assert result["supportable_ue_counts"]["heavy_8mbps"] == 5
    assert "oversubscribed" in result["saturation_at_heavy_demand"]


# ─────────────────────────────────────────────────────────────
# Module 8 — Temporal Urgency
# ─────────────────────────────────────────────────────────────

def test_temporal_urgency_silent_onset_is_critical():
    result = call(get_temporal_urgency, pattern="silent_onset")
    assert result["urgency"] == "CRITICAL"
    assert result["is_real_fault"] is True

def test_temporal_urgency_diurnal_drift_is_not_a_fault():
    # diurnal_drift is load-correlated, explicitly NOT a real fault.
    result = call(get_temporal_urgency, pattern="diurnal_drift")
    assert result["is_real_fault"] is False
    assert result["urgency"] == "NONE"


# ─────────────────────────────────────────────────────────────
# Step 8 — A1 Policy Publisher
# ─────────────────────────────────────────────────────────────

def test_generate_a1_policy_has_required_fields():
    policy = call(generate_a1_policy,
                  target_cell="cell_A", actual_aggressor="cell_C",
                  cause="interference", parameter_target="tiltDelta",
                  parameter_delta=2.0)
    for field in ("policyId", "policyClass", "targetCell",
                  "parameterTargets", "provenance", "risk"):
        assert field in policy
    assert policy["risk"]["compositeRiskScore"] is not None

def test_generate_a1_policy_capacity_cause_requires_approval():
    # A capacity-cause policy must flag human approval — the HITL guardrail.
    policy = call(generate_a1_policy,
                  target_cell="cell_A", actual_aggressor="cell_A",
                  cause="capacity", parameter_target="txPowerDelta",
                  parameter_delta=0.0)
    assert policy["requiresApproval"] is True

def test_generate_a1_policy_interference_cause_does_not_require_approval():
    policy = call(generate_a1_policy,
                  target_cell="cell_A", actual_aggressor="cell_C",
                  cause="interference", parameter_target="tiltDelta",
                  parameter_delta=1.5)
    assert policy["requiresApproval"] is False
    assert policy["policyClass"] == "ORAN_QOS_1"


# ─────────────────────────────────────────────────────────────
# Tool registry wiring
# ─────────────────────────────────────────────────────────────

def test_each_agent_registry_is_non_empty():
    for registry in (DIAGNOSTICIAN_TOOLS, INTERFERENCE_TOOLS,
                      CAPACITY_TOOLS, POLICY_WRITER_TOOLS):
        assert len(registry) > 0

def test_all_tools_is_the_union_of_agent_registries():
    expected = (len(DIAGNOSTICIAN_TOOLS) + len(INTERFERENCE_TOOLS)
                + len(CAPACITY_TOOLS) + len(POLICY_WRITER_TOOLS))
    assert len(ALL_TOOLS) == expected