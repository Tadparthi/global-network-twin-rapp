"""
Pre-computed outputs from the GNT modules.

In a production deployment these tools would call the trained ML models directly.
For this portfolio piece, we use the actual JSON outputs produced by your GNT build —
this means the agent system is grounded in real model outputs, not fabricated data,
without requiring you to ship the trained models.
"""

# From sprint1_benchmark_report.html / benchmark.json
M1_SINR_BASELINE = {
    "cell_A": {"mean_sinr_db": 5.7, "p5_sinr_db": -2.3, "p95_sinr_db": 14.1, "pct_poor": 41.0, "n_ues": 244},
    "cell_B": {"mean_sinr_db": 8.2, "p5_sinr_db": 1.4,  "p95_sinr_db": 16.3, "pct_poor": 12.0, "n_ues": 136},
    "cell_C": {"mean_sinr_db": 6.6, "p5_sinr_db": -0.8, "p95_sinr_db": 15.2, "pct_poor": 28.0, "n_ues": 260},
}
M1_RMSE_BOUNDS = {"in_distribution": 0.573, "out_of_distribution": 2.09, "bias_db": 0.02}

# From module2_v2_results.json — directional ISR matrix (mean over 400 perturbations)
M2_COUPLING = {
    "cell_A->cell_B": 0.221,
    "cell_A->cell_C": 0.221,
    "cell_B->cell_A": 0.137,
    "cell_B->cell_C": 0.119,
    "cell_C->cell_A": 0.223,   # dominant aggressor path
    "cell_C->cell_B": 0.203,
}
M2_BLAME = {"cell_A": 0.327, "cell_B": 0.354, "cell_C": 0.319}

# From module3_coverage_capacity.json
M3_VERDICTS = {
    "cell_A_at_8mbps": {"class": "CAPACITY_BOUND", "confidence": 0.94, "cong_ratio": 5.2},
    "cell_B_at_8mbps": {"class": "CAPACITY_BOUND", "confidence": 0.88, "cong_ratio": 1.9},
    "cell_C_at_8mbps": {"class": "CAPACITY_BOUND", "confidence": 0.91, "cong_ratio": 2.3},
    "cell_A_at_3mbps": {"class": "COVERAGE_BOUND", "confidence": 0.81, "cong_ratio": 1.1},
}

# From sprint2_gap_diagnostics — mobility classifier
M4_MOBILITY = {
    "stationary":  {"f1": 0.942, "speed_threshold_mps": 0.8},
    "slow_moving": {"f1": 0.933, "speed_range_mps": [0.8, 2.5]},
    "fast_moving": {"f1": 0.992, "speed_threshold_mps": 2.5},
}

# From module7_capacity_planning.json — top NPV intervention
M7_TOP_INTERVENTION = {
    "scenario_id": "SCN-0383",
    "sinr_gain_db": 3.39,
    "intervention_cost_gbp": 8500,
    "annual_revenue_uplift_gbp": 18463,
    "npv_3yr_gbp": 37416,
    "roic_pct": 117,
    "payback_months": 5.5,
    "intervention_type": "azimuth_change",
    "target_cell": "cell_C",
}

# From module7_capacity_planning.json — supportable UE counts
M7_SUPPORTABLE_UES = {
    "cell_A": {"light_1mbps": 50, "moderate_3mbps": 25, "heavy_8mbps": 5,  "peak_15mbps": 2},
    "cell_B": {"light_1mbps": 77, "moderate_3mbps": 27, "heavy_8mbps": 14, "peak_15mbps": 7},
    "cell_C": {"light_1mbps": 94, "moderate_3mbps": 24, "heavy_8mbps": 12, "peak_15mbps": 9},
}

# From module8_temporal_anomaly.json — pattern-to-urgency mapping
M8_URGENCY = {
    "slow_burn":           {"urgency": "MEDIUM",   "window": "3-7 days", "cause": "PA degradation, antenna corrosion"},
    "silent_onset":        {"urgency": "CRITICAL", "window": "4 hours",   "cause": "hardware imminent failure"},
    "instantaneous_step":  {"urgency": "HIGH",     "window": "1 hour",    "cause": "config push, link cut"},
    "logistic_s_curve":    {"urgency": "MEDIUM",   "window": "24 hours",  "cause": "load saturation"},
    "partial_degradation": {"urgency": "LOW",      "window": "1 week",    "cause": "sector-specific issue"},
    "diurnal_drift":       {"urgency": "NONE",     "window": "n/a",       "cause": "load-correlated, not a fault"},
    "compound_nonlinear":  {"urgency": "HIGH",     "window": "4 hours",   "cause": "multiple concurrent issues"},
}
