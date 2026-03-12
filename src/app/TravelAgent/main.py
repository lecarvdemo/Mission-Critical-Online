"""
Travel Agent Orchestrator
=========================
Entry point that runs all three agents in sequence and saves the outputs:

  1. TravelPlanningAgent  → generates the Europe itinerary
  2. FlightPlanningAgent  → generates the flight plan from Canberra
  3. ReviewAgent          → reviews both plans and produces the final summary

Usage
-----
    # Using OpenAI
    export OPENAI_API_KEY="sk-..."
    python main.py

    # Using Azure OpenAI
    export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
    export AZURE_OPENAI_API_KEY="<key>"
    export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
    python main.py

    # Optional: save results to a file
    python main.py --output results/trip_plan.md

    # Optional: load a .env file for credentials
    python main.py --env .env
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from travel_planning_agent import create_agent as create_travel_agent
from flight_planning_agent import create_agent as create_flight_agent
from review_agent import create_agent as create_review_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEPARATOR = "=" * 80


def _section(title: str, content: str) -> str:
    lines = [SEPARATOR, f"  {title}", SEPARATOR, "", content, ""]
    return "\n".join(lines)


def _banner(message: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {message}")
    print(f"{'─' * 60}\n", flush=True)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run(output_path: str | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- 1. Travel Planning Agent ------------------------------------------
    _banner("Running Travel Planning Agent …")
    travel_agent = create_travel_agent()
    travel_plan = travel_agent.run()
    print(travel_plan)

    # -- 2. Flight Planning Agent ------------------------------------------
    _banner("Running Flight Planning Agent …")
    flight_agent = create_flight_agent()
    flight_plan = flight_agent.run()
    print(flight_plan)

    # -- 3. Review Agent ---------------------------------------------------
    _banner("Running Review Agent (synthesising both plans) …")
    review_agent = create_review_agent()
    final_summary = review_agent.run(
        travel_plan=travel_plan,
        flight_plan=flight_plan,
    )
    print(final_summary)

    # -- Combine all outputs -----------------------------------------------
    full_report = "\n".join(
        [
            f"# Europe Winter Trip – December 2027",
            f"_Generated: {timestamp}_",
            "",
            _section("TRAVEL PLANNING AGENT", travel_plan),
            _section("FLIGHT PLANNING AGENT", flight_plan),
            _section("REVIEW AGENT – FINAL SUMMARY", final_summary),
        ]
    )

    # -- Save to file (optional) -------------------------------------------
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(full_report, encoding="utf-8")
        print(f"\n✅  Full report saved to: {out.resolve()}")
    else:
        _banner("Full Report")
        print(full_report)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Run the three travel-agent pipeline (Travel → Flight → Review)
            and optionally save the combined report to a Markdown file.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Path to save the combined Markdown report (optional).",
    )
    parser.add_argument(
        "--env",
        metavar="FILE",
        default=None,
        help="Path to a .env file with API credentials (optional).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load .env file if provided
    if args.env:
        load_dotenv(args.env)
    else:
        # Try to auto-load a .env in the current or script directory
        for candidate in [Path(".env"), Path(__file__).parent / ".env"]:
            if candidate.exists():
                load_dotenv(candidate)
                break

    # Validate that at least one API credential source is configured
    has_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if not has_azure and not has_openai:
        print(
            "ERROR: No API credentials found.\n"
            "Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY  (Azure OpenAI)\n"
            "or OPENAI_API_KEY  (OpenAI).",
            file=sys.stderr,
        )
        sys.exit(1)

    run(output_path=args.output)


if __name__ == "__main__":
    main()
