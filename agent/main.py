"""Command-line entry point for the MLOps agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agent.envfile import load_dotenv
from agent.orchestrator import Agent

SCENARIOS = [
    "Which dataset version is recorded, and how do I reproduce it?",
    "Compare the last five training runs and recommend the best candidate based on F1 score.",
    "Trigger the model training pipeline and report whether execution succeeded.",
    "Why is my prediction API unavailable?",
    "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ask the MLOps agent a question.")
    parser.add_argument("message", nargs="*")
    parser.add_argument("--scenarios", action="store_true", help="Run the five graded conversations.")
    parser.add_argument("--out", default="evidence/demo-scenarios.json")
    args = parser.parse_args(argv)
    agent = Agent()
    if args.scenarios:
        payload = []
        for message in SCENARIOS:
            response = agent.handle(message)
            payload.append(response.model_dump())
            print(response.answer)
            print("\n----\n")
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    if not args.message:
        parser.error("provide a message or --scenarios")
    response = agent.handle(" ".join(args.message))
    print(response.answer)
    return 0 if response.facts or response.uncertainties else 1


if __name__ == "__main__":
    sys.exit(main())
