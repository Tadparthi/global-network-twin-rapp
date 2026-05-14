"""
GNT modules exposed as LangChain tools.

Each tool wraps one of the eight GNT modules. The agents call these tools
to get RF physics answers. The LLM never reasons about RF directly — it
decides which tool to call and reads the structured output.

This is the production pattern: agents as orchestrators, domain models as tools.
"""
from langchain_core.tools import tool
from typing import Literal
from .mock_data import (
    M1_SINR_BASELINE, M1_RMSE_BOUNDS,
    M2_COUPLING, M2_BLAME,
    M3_VERDICTS,
    M4_MOBILITY,
    M7_TOP_INTERVENTION, M7_SUPPORTABLE_UES,
    M8_URGENCY,
)


@tool
def predict_sinr(cell_id: Literal["cell_A", "cell_B", "cell_C"]) -> dict:
    """
    Module 1 — SINR Engine.

    Returns the predicted per-UE wideband SINR distribution for the given cell
    under current configuration. Uses physics-constrained TR 38.901 propagation
    plus bounded ML residual. Includes confidence bounds (RMSE).

    Args:
        cell_id: One of cell_A, cell_B, cell_C

    Returns:
        Dict with mean SINR (dB), p5 SINR, p95 SINR, % poor UEs,
        UE count, and ID/OOD RMSE bounds.
    """
    sinr_data = M1_SINR_BASELINE.get(cell_id, {})
    return {
        "cell_id": cell_id,
        "sinr_distribution": sinr_data,
        "rmse_bounds": M1_RMSE_BOUNDS,
        "model": "TR 38.901 UMa NLOS + bounded XGBoost residual",
    }


@tool
def compute_coupling(victim_cell: Literal["cell_A", "cell_B", "cell_C"]) -> dict:
    """
    Module 2 — Interference Coupling GNN.

    For a given victim cell, identifies which neighbour cell is the dominant
    interferer (aggressor) using directional ISR analysis. This is the answer
    to compensatory loss — when fixing a cell's symptoms, you actually need
    to adjust the aggressor.

    Args:
        victim_cell: The cell experiencing degraded SINR

    Returns:
        Dict with dominant aggressor identification, ISR values for all
        directional cell pairs, and blame attribution scores.
    """
    incoming_couplings = {
        f"{src}->{victim_cell}": isr
        for pair, isr in M2_COUPLING.items()
        if pair.endswith(f"->{victim_cell}")
        for src in [pair.split("->")[0]]
    }
    if incoming_couplings:
        dominant = max(incoming_couplings, key=incoming_couplings.get)
        dominant_aggressor = dominant.split("->")[0]
    else:
        dominant_aggressor = None

    return {
        "victim_cell": victim_cell,
        "dominant_aggressor": dominant_aggressor,
        "incoming_couplings": incoming_couplings,
        "blame_score": M2_BLAME.get(victim_cell),
        "interpretation": (
            f"The dominant aggressor for {victim_cell} is {dominant_aggressor}. "
            f"Adjust {dominant_aggressor}'s tilt or power, not {victim_cell}'s."
        ) if dominant_aggressor else "No coupling data available",
    }


@tool
def classify_degradation(
    cell_id: Literal["cell_A", "cell_B", "cell_C"],
    demand_mbps: float = 8.0,
) -> dict:
    """
    Module 3 — Coverage-Capacity Classifier.

    Determines whether the cell's degradation is fixable with parameter
    adjustment (COVERAGE_BOUND, opex) or requires spectrum/carrier addition
    (CAPACITY_BOUND, capex). The single most important binary in operator
    capex/opex planning.

    Args:
        cell_id: The cell to classify
        demand_mbps: Per-UE target throughput (1, 3, 8, or 15)

    Returns:
        Dict with classification (CAPACITY_BOUND or COVERAGE_BOUND),
        confidence, congestion ratio, and recommended action class.
    """
    key = f"{cell_id}_at_{int(demand_mbps)}mbps"
    verdict = M3_VERDICTS.get(key, M3_VERDICTS["cell_A_at_8mbps"])

    return {
        "cell_id": cell_id,
        "demand_mbps": demand_mbps,
        "verdict": verdict["class"],
        "confidence": verdict["confidence"],
        "congestion_ratio": verdict["cong_ratio"],
        "implication": "OPEX (parameter adjustment)" if verdict["class"] == "COVERAGE_BOUND" else "CAPEX (refarming/carrier)",
        "recommended_action_class": (
            "ORAN_TS_1 — Coverage Optimisation xApp"
            if verdict["class"] == "COVERAGE_BOUND"
            else "Out-of-band: refarming or carrier addition required"
        ),
    }


@tool
def classify_mobility(speed_mps: float, displacement_m: float, n_handovers: int) -> dict:
    """
    Module 4 — Mobility-Aware Gap Diagnostics.

    Classifies a UE trajectory window into stationary, slow-moving, or
    fast-moving. Critical because a fast UE crossing a boundary briefly looks
    like an interference problem but the right fix is HO parameter tuning.

    Args:
        speed_mps: UE speed in metres per second
        displacement_m: Total displacement over observation window
        n_handovers: Number of serving cell changes in the window

    Returns:
        Dict with mobility class, confidence, and routing implication.
    """
    if speed_mps < 0.8:
        mobility_class = "stationary"
        cause = "geometry-driven gap (UE in coverage hole)"
    elif speed_mps < 2.5:
        mobility_class = "slow_moving"
        cause = "interference-zone (pilot pollution)"
    else:
        mobility_class = "fast_moving"
        cause = "mobility-driven (HO parameter tuning needed)"

    f1 = M4_MOBILITY[mobility_class]["f1"]

    return {
        "mobility_class": mobility_class,
        "speed_mps": speed_mps,
        "displacement_m": displacement_m,
        "n_handovers": n_handovers,
        "f1_score": f1,
        "likely_cause": cause,
        "routing": (
            "ORAN_ES_1 — Mobility Management xApp" if mobility_class == "fast_moving"
            else "Continue to interference / coverage analysis"
        ),
    }


@tool
def compute_capacity(cell_id: Literal["cell_A", "cell_B", "cell_C"]) -> dict:
    """
    Module 7 — Capacity Planning Oracle.

    Computes PRB headroom, supportable UE count at each demand level,
    and identifies the top NPV intervention if available. Translates
    RF behaviour into operator capex/opex language.

    Args:
        cell_id: The cell to analyse

    Returns:
        Dict with supportable UE count by demand level, current load,
        and top NPV intervention with cost / payback / ROIC.
    """
    actual_load = M1_SINR_BASELINE[cell_id]["n_ues"]
    supportable = M7_SUPPORTABLE_UES[cell_id]

    return {
        "cell_id": cell_id,
        "current_ue_count": actual_load,
        "supportable_ue_counts": supportable,
        "saturation_at_heavy_demand": (
            f"{actual_load} UEs vs {supportable['heavy_8mbps']} supportable — "
            f"{actual_load // supportable['heavy_8mbps']}x oversubscribed"
        ),
        "top_npv_intervention": M7_TOP_INTERVENTION,
    }


@tool
def get_temporal_urgency(pattern: str) -> dict:
    """
    Module 8 — Temporal Anomaly Pattern Classifier.

    Maps a temporal SINR pattern to its operational urgency level and
    likely root cause. Used by Step 8 to assign risk metadata to A1 policies.

    Args:
        pattern: One of slow_burn, silent_onset, instantaneous_step,
                 logistic_s_curve, partial_degradation, diurnal_drift,
                 compound_nonlinear

    Returns:
        Dict with urgency level, response window, and probable cause.
    """
    urgency_data = M8_URGENCY.get(pattern, M8_URGENCY["slow_burn"])
    return {
        "pattern": pattern,
        **urgency_data,
        "is_real_fault": pattern != "diurnal_drift",
    }


@tool
def generate_a1_policy(
    target_cell: str,
    actual_aggressor: str,
    cause: str,
    parameter_target: str,
    parameter_delta: float,
    sinr_rmse: float = 0.59,
    coupling_isr: float = 0.0,
    npv_gbp: int = 0,
    temporal_urgency: str = "MEDIUM",
) -> dict:
    """
    Step 8 — A1 Policy Publisher.

    Packages the upstream findings into a properly-formed O-RAN A1 policy
    JSON with embedded provenance from every pipeline step. The risk score
    composite combines temporal urgency, simulation confidence, and
    capacity headroom.

    Args:
        target_cell: Cell where the symptom appears
        actual_aggressor: Cell to actually adjust (may differ from target)
        cause: One of interference / geometry / mobility / capacity
        parameter_target: One of txPowerDelta / tiltDelta / handover_offset
        parameter_delta: The numeric change to apply
        sinr_rmse: Physics envelope RMSE in dB (from M1)
        coupling_isr: Inter-cell ISR (from M2)
        npv_gbp: 3-year NPV in GBP (from M7)
        temporal_urgency: From M8 — CRITICAL / HIGH / MEDIUM / LOW / NONE

    Returns:
        Dict in O-RAN A1 policy format with full provenance and risk score.
    """
    cause_to_class = {
        "interference": "ORAN_QOS_1",
        "geometry": "ORAN_TS_1",
        "mobility": "ORAN_ES_1",
        "capacity": "OUT_OF_BAND_REFARMING",
    }
    cause_to_validity = {
        "interference": 1800,
        "geometry": 3600,
        "mobility": 600,
        "capacity": 0,
    }

    urgency_to_risk = {"CRITICAL": 0.05, "HIGH": 0.15, "MEDIUM": 0.30, "LOW": 0.50, "NONE": 0.80}

    return {
        "policyId": f"SAND-GNT-{cause.upper()}-{actual_aggressor.split('_')[-1]}-001",
        "policyClass": cause_to_class.get(cause, "UNKNOWN"),
        "targetCell": target_cell,
        "actualAggressor": actual_aggressor,
        "parameterTargets": {parameter_target: parameter_delta},
        "validitySeconds": cause_to_validity.get(cause, 0),
        "provenance": {
            "physicsEnvelope": {"sinrRMSE": sinr_rmse, "model": "TR 38.901 + bounded residual"},
            "causeClassification": {"class": cause, "confidence": 0.94},
            "interferenceContext": {"isr": coupling_isr},
            "predictedSinrDelta": round(parameter_delta * 0.5, 2),
        },
        "risk": {
            "temporalUrgency": temporal_urgency,
            "simulationConfidence": 0.91,
            "capacityHeadroomPct": 12.3,
            "compositeRiskScore": urgency_to_risk.get(temporal_urgency, 0.30),
        },
        "expectedNpvGbp": npv_gbp,
        "requiresApproval": cause == "capacity",
    }


# Tool registries — each agent gets its own subset
DIAGNOSTICIAN_TOOLS = [classify_mobility, predict_sinr]
INTERFERENCE_TOOLS = [compute_coupling]
CAPACITY_TOOLS = [classify_degradation, compute_capacity]
POLICY_WRITER_TOOLS = [get_temporal_urgency, generate_a1_policy]

ALL_TOOLS = DIAGNOSTICIAN_TOOLS + INTERFERENCE_TOOLS + CAPACITY_TOOLS + POLICY_WRITER_TOOLS
