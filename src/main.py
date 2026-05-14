"""
Entry point. Run a scenario through the multi-agent system.

Usage:
    python -m src.main --scenario congested_cell
    python -m src.main --scenario interference_spike --approve-policy
    python -m src.main --scenario mobility_overload

The --approve-policy flag enables human-in-the-loop mode: the graph
pauses before the policy_writer node, prints the upstream findings,
and waits for operator approval (y/n) before generating the A1 policy.

This is the production pattern for operator-supervised RF intervention
workflows — the agentic equivalent of the GNT Daily Brief approval step.
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from src.graph import build_graph


def load_scenario(name: str) -> dict:
    """Load a scenario JSON from the scenarios/ directory."""
    scenario_path = Path(__file__).parent.parent / "scenarios" / f"{name}.json"
    if not scenario_path.exists():
        print(f"ERROR: scenario file not found: {scenario_path}")
        print(f"Available scenarios: {[p.stem for p in scenario_path.parent.glob('*.json')]}")
        sys.exit(1)
    with open(scenario_path) as f:
        return json.load(f)


def _short(s, max_len=400):
    """Truncate long strings for console display."""
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + "..."


def print_findings_for_review(state: dict) -> None:
    """Print the upstream findings so the operator can review before approving."""
    print("\n" + "=" * 70)
    print("  HUMAN-IN-THE-LOOP CHECKPOINT — REVIEW BEFORE POLICY GENERATION")
    print("=" * 70)

    diag = state.get("diagnostic_findings", {})
    if diag.get("diagnosis"):
        print("\n[DIAGNOSTICIAN]")
        print(_short(diag["diagnosis"]))

    interf = state.get("interference_findings", {})
    if interf.get("analysis"):
        print("\n[INTERFERENCE ANALYST]")
        print(_short(interf["analysis"]))

    cap = state.get("capacity_findings", {})
    if cap.get("verdict"):
        print("\n[CAPACITY PLANNER]")
        print(_short(cap["verdict"]))

    print("\n" + "=" * 70)


def run_scenario(scenario_name: str, model: str, approve_policy: bool) -> None:
    """Run one scenario through the multi-agent system."""
    scenario = load_scenario(scenario_name)

    print("=" * 70)
    print(f"  RF MULTI-AGENT SYSTEM — Scenario: {scenario_name}")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Description: {scenario.get('description', '')[:80]}")
    print(f"  Mode: {'human-in-the-loop' if approve_policy else 'autonomous end-to-end'}")
    print("=" * 70)

    graph = build_graph(model_name=model, human_in_the_loop=approve_policy)

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

    # In human-in-the-loop mode we need a thread_id so the checkpointer
    # can save/restore state across the two invocations
    config = {"recursion_limit": 25}
    if approve_policy:
        thread_id = f"scenario-{uuid.uuid4().hex[:8]}"
        config["configurable"] = {"thread_id": thread_id}

    # First invocation — runs until either END (no interrupt) or the interrupt
    final_state = graph.invoke(initial_state, config=config)

    # If we're in human-in-the-loop mode and the graph hit the interrupt,
    # final_state won't yet contain the policy. We need to approve and resume.
    if approve_policy and not final_state.get("policy_output", {}).get("summary"):
        print_findings_for_review(final_state)

        while True:
            response = input("\nApprove A1 policy generation? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                print("\n[OPERATOR] Approved. Resuming graph execution.\n")
                # Resume by invoking with None — checkpointer restores state
                final_state = graph.invoke(None, config=config)
                break
            elif response in ("n", "no"):
                print("\n[OPERATOR] Rejected. Graph halted before policy generation.")
                print("\n" + "=" * 70)
                print("  FINDINGS (no policy generated)")
                print("=" * 70)
                print(_short(final_state.get("diagnostic_findings", {}).get("diagnosis", ""), 600))
                return
            else:
                print("  Please enter y or n.")

    # Print the final recommendation
    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATION")
    print("=" * 70)
    print(final_state.get("final_recommendation", "(no recommendation generated)"))
    print("\n" + "=" * 70)

    # Print the A1 policy if generated
    policy_output = final_state.get("policy_output", {})
    a1_policy = policy_output.get("tool_outputs", {}).get("generate_a1_policy")
    if a1_policy:
        print("  GENERATED A1 POLICY")
        print("=" * 70)
        print(json.dumps(a1_policy, indent=2))
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run a scenario through the RF multi-agent system")
    parser.add_argument(
        "--scenario", default="congested_cell",
        help="Name of scenario file (without .json) in scenarios/ directory"
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-5",
        help="Anthropic model name"
    )
    parser.add_argument(
        "--approve-policy", action="store_true",
        help="Enable human-in-the-loop mode: pause before policy generation and wait for operator approval"
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with: $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)")
        sys.exit(1)

    run_scenario(args.scenario, args.model, args.approve_policy)


if __name__ == "__main__":
    main()
