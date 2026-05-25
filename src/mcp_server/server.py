import os
from typing import Any, Dict, List

import requests
from fastmcp import FastMCP


DEFAULT_GNT_API_BASE_URL = "https://global-network-twin-rapp.onrender.com"

GNT_API_BASE_URL = os.getenv(
    "GNT_API_BASE_URL",
    DEFAULT_GNT_API_BASE_URL,
).rstrip("/")


mcp = FastMCP("Global Network Twin MCP Server")


@mcp.tool()
def health_check_gnt_api() -> Dict[str, Any]:
    """
    Check whether the Global Network Twin FastAPI backend is reachable.
    """
    response = requests.get(f"{GNT_API_BASE_URL}/health", timeout=30)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def list_gnt_scenarios() -> Dict[str, Any]:
    """
    List available Global Network Twin analysis scenarios.
    """
    response = requests.get(f"{GNT_API_BASE_URL}/scenarios", timeout=30)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def analyze_gnt_scenario(
    scenario_name: str,
    model: str = "claude-sonnet-4-5",
    include_debug: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a Global Network Twin scenario using the deployed multi-agent rApp.

    Args:
        scenario_name: Scenario name such as congested_cell, interference_spike, or mobility_overload.
        model: Anthropic model name used by the backend agent workflow.
        include_debug: If true, include raw agent/tool outputs from the backend.
    """
    payload = {
        "scenario_name": scenario_name,
        "model": model,
        "include_debug": include_debug,
    }

    response = requests.post(
        f"{GNT_API_BASE_URL}/analyze-scenario",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run()