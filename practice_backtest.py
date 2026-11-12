#!/usr/bin/env python3
"""Reproduce chronological practice scoring against a simple starter baseline.

This tool uses only the supplied kit files and Python's standard library. It is
for local practice comparisons; its scores are not the private official grader.
Run it from the repository root with:

    python3 practice_backtest.py --kit /path/to/pumpfun-top5-forecast-01-kit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import main as engine


def score_forecast(
    forecast: dict[str, Any],
    target: dict[str, Any],
    edges: list[float],
) -> tuple[float, float, int]:
    """Return order log score, size-bin log score, and top-five agreement."""
    chart = target["rows"][:100]
    chart_mints = [row["mint"] for row in chart]
    order_score = 0.0
    size_score = 0.0

    for place in range(1, engine.PLACES + 1):
        actual = chart[place - 1]
        reported = {
            row["mint"]: row["p"]
            for row in forecast["probabilities"]
            if row["place"] == place
        }
        named_mass = sum(reported.values())
        if actual["mint"] in reported:
            coin_probability = reported[actual["mint"]]
        else:
            wildcard_count = sum(mint not in reported for mint in chart_mints)
            coin_probability = max(0.0, 1.0 - named_mass) / max(1, wildcard_count)
        order_score += math.log(max(coin_probability, 1e-4))

        actual_bin = engine.log_bin(engine.log_cap(actual), edges)
        size_probability = next(
            (
                row["p"]
                for row in forecast["sizes"]
                if row["place"] == place and row["bin"] == actual_bin
            ),
            0.0,
        )
        size_score += math.log(max(size_probability, 1e-4))

    agreement = 0
    actual_top = chart_mints[:engine.PLACES]
    for forecast_position, mint in enumerate(forecast["top5"]):
        if mint in actual_top:
            actual_position = actual_top.index(mint)
            agreement += engine.PLACES - abs(forecast_position - actual_position)
    return order_score, size_score, agreement


def starter_baseline(
    cut: dict[str, Any],
    edges: list[float],
) -> dict[str, Any]:
    """Forecast from current rank and a narrow normal size distribution."""
    chart = cut["rows"]
    probabilities = []
    sizes = []
    top5 = [row["mint"] for row in chart[:engine.PLACES]]

    for place in range(1, engine.PLACES + 1):
        center = place - 1
        probabilities.append({"mint": chart[center]["mint"], "place": place, "p": 0.30})
        if center > 0:
            probabilities.append({"mint": chart[center - 1]["mint"], "place": place, "p": 0.10})
        if center + 1 < len(chart):
            probabilities.append({"mint": chart[center + 1]["mint"], "place": place, "p": 0.10})
        cap_mean = engine.log_cap(chart[center])
        for bin_index, probability in enumerate(engine.normal_bins(cap_mean, 0.35, edges)):
            sizes.append({"place": place, "bin": bin_index, "p": probability})

    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def rolling_cases(
    task: dict[str, Any],
    marks: list[dict[str, Any]],
    max_horizon_marks: int,
) -> list[dict[str, Any]]:
    """Forecast from each chronological cut and score every later mark."""
    cases = []
    for cut_index in range(1, len(marks) - 1):
        stop = min(len(marks), cut_index + 1 + max_horizon_marks)
        future = marks[cut_index + 1 : stop]
        future = [
            target for target in future
            if len(target["rows"]) >= engine.PLACES
        ]
        if not future:
            continue

        cut_task = dict(task)
        cut_task["history_until"] = marks[cut_index]["at"].isoformat()
        cut_task["reads"] = [mark["at"].isoformat() for mark in future]
        visible_history = marks[: cut_index + 1]
        prediction = engine.forecast(cut_task, visible_history)
        baseline = starter_baseline(marks[cut_index], task["size"]["edges"])

        for target in future:
            model_scores = score_forecast(prediction, target, task["size"]["edges"])
            base_scores = score_forecast(baseline, target, task["size"]["edges"])
            cases.append(
                {
                    "target_at": target["at"].isoformat(),
                    "model": model_scores,
                    "baseline": base_scores,
                }
            )
    return cases


def mean(values: list[float]) -> float:
    """Use a clear zero result when a requested summary has no cases."""
    return sum(values) / len(values) if values else 0.0


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case scores without hiding the case count."""
    result: dict[str, Any] = {"cases": len(cases)}
    for label in ("model", "baseline"):
        result[label] = {
            "order_log_score": round(mean([case[label][0] for case in cases]), 6),
            "size_log_score": round(mean([case[label][1] for case in cases]), 6),
            "combined_log_score": round(
                mean([case[label][0] + case[label][1] for case in cases]), 6
            ),
            "top5_agreement": sum(case[label][2] for case in cases),
        }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kit",
        type=Path,
        required=True,
        help="unpacked practice kit directory containing input/",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=engine.SIMULATIONS,
        help="Monte Carlo draws per forecast (default: %(default)s)",
    )
    parser.add_argument(
        "--max-horizon-marks",
        type=int,
        default=5,
        help="maximum later marks scored from each chronological cut",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="include each target's model and baseline scores",
    )
    args = parser.parse_args(argv)
    if args.simulations < 1:
        parser.error("--simulations must be positive")
    if args.max_horizon_marks < 1:
        parser.error("--max-horizon-marks must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.kit / "input"
    if not (input_dir / "task.json").is_file() or not (input_dir / "history.ndjson").is_file():
        print(f"practice input files are missing under {input_dir}", file=sys.stderr)
        return 2

    try:
        task, marks = engine.load_history(input_dir)
        engine.validate_inputs(task, marks)
        engine.SIMULATIONS = args.simulations
        cases = rolling_cases(task, marks, args.max_horizon_marks)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"could not backtest practice data: {exc}", file=sys.stderr)
        return 2

    output = summarize(cases)
    if args.details:
        output["details"] = cases
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
