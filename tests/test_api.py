"""
API-layer tests for the GNT multi-agent rApp (src/api/main.py).

/health and /scenarios are pure and need no API key.
The /analyze-scenario test checks the error path only — it asserts the
service fails cleanly when ANTHROPIC_API_KEY is absent, rather than
spending tokens on a full LLM run in CI.

Run:  pytest tests/test_api.py -v
"""
import os
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint_reports_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_scenarios_endpoint_lists_known_scenarios():
    response = client.get("/scenarios")
    assert response.status_code == 200
    scenarios = response.json()["scenarios"]
    # The three shipped scenario files must be discoverable.
    for name in ("congested_cell", "interference_spike", "mobility_overload"):
        assert name in scenarios


def test_analyze_scenario_fails_cleanly_without_api_key(monkeypatch):
    # With no API key, the endpoint must return a clean 500 with a clear
    # message — not an unhandled crash. This is a test *of* the error handling.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post("/analyze-scenario",
                           json={"scenario_name": "congested_cell"})
    assert response.status_code == 500
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_analyze_scenario_unknown_name_returns_404(monkeypatch):
    # An unknown scenario name should 404 with the available list —
    # but only once past the API-key check, so set a dummy key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-routing-test")
    response = client.post("/analyze-scenario",
                           json={"scenario_name": "does_not_exist"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "available_scenarios" in detail