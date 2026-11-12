#!/usr/bin/env python3
"""Deterministic, offline top-five and market-cap forecaster for make.fun."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PLACES = 5
SIMULATIONS = 1800
MAX_CANDIDATES = 230
MAX_ROWS_PER_PLACE = 100
VOLATILITY_SCALE = 2.4
CRASH_HAZARD = 0.038
LAUNCH_HAZARD = 0.026
UNKNOWN_SIZE_SD = 0.90
UNKNOWN_SIZE_HORIZON_SD = 0.018
UNKNOWN_SIZE_GROWTH_PER_HOUR = 0.035
KNOWN_PRIOR_START = 0.56
KNOWN_PRIOR_FADE = 0.012
MOMENTUM_SCALE = 0.08


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def log_cap(row: dict) -> float:
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def bounded(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def log_bin(value: float, edges: list[float]) -> int:
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def normal_cdf(value: float, mean: float, sd: float) -> float:
    return 0.5 * (1.0 + math.erf((value - mean) / (sd * math.sqrt(2.0))))


def normal_bins(mean: float, sd: float, edges: list[float]) -> list[float]:
    result = [0.0] * (len(edges) + 1)
    result[0] = normal_cdf(edges[0], mean, sd)
    for i in range(1, len(edges)):
        result[i] = normal_cdf(edges[i], mean, sd) - normal_cdf(edges[i - 1], mean, sd)
    result[-1] = 1.0 - normal_cdf(edges[-1], mean, sd)
    total = sum(result)
    if total <= 0.0:
        result[log_bin(mean, edges)] = 1.0
    else:
        result = [max(0.0, p / total) for p in result]
    return result


def load_history(input_dir: Path) -> tuple[dict, list[dict]]:
    task = json.loads((input_dir / "task.json").read_text(encoding="utf-8"))
    marks = []
    for line in (input_dir / "history.ndjson").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        mark = json.loads(line)
        if mark.get("ok") is True and isinstance(mark.get("rows"), list):
            rows = sorted(mark["rows"], key=lambda row: int(row.get("rank", 10**9)))
            marks.append({"at": parse_time(mark["at"]), "rows": rows})
    if not marks:
        raise ValueError("history.ndjson has no valid marks")
    if len(marks[-1]["rows"]) < PLACES:
        raise ValueError("the last mark ranks fewer than five coins")
    return task, marks


def build_candidates(marks: list[dict], cadence_minutes: int) -> tuple[list[dict], dict[str, dict]]:
    histories: dict[str, list[tuple[int, dict, float]]] = defaultdict(list)
    for mark_index, mark in enumerate(marks):
        for row in mark["rows"]:
            histories[row["mint"]].append((mark_index, row, log_cap(row)))

    last_index = len(marks) - 1
    latest = {row["mint"]: row for row in marks[-1]["rows"]}
    selected = set(latest)
    for mint, records in histories.items():
        best_rank = min(int(record[1]["rank"]) for record in records)
        if best_rank <= 15 and last_index - records[-1][0] <= 2:
            selected.add(mint)

    candidates = []
    for mint in selected:
        records = histories[mint]
        index, row, current_log = records[-1]
        stale = last_index - index
        if stale:
            current_log -= 0.16 * stale

        changes = []
        for left, right in zip(records, records[1:]):
            gap = right[0] - left[0]
            if gap > 0:
                changes.append((right[2] - left[2]) / gap)
        recent = changes[-1] if changes else 0.0
        medium = sum(changes[-3:]) / max(1, len(changes[-3:]))
        momentum = 0.65 * recent + 0.35 * medium
        try:
            age_hours = max(0.0, (marks[-1]["at"] - parse_time(row["created_at"])).total_seconds() / 3600.0)
        except (KeyError, ValueError):
            age_hours = 0.5

        candidates.append({
            "mint": mint,
            "base": current_log,
            "rank": int(row["rank"]),
            "age": age_hours,
            "stale": stale,
            "momentum": bounded(momentum, -0.75, 0.75),
            "last_cap": log_cap(row),
        })

    candidates.sort(key=lambda item: (item["rank"], -item["base"], item["mint"]))
    return candidates[:MAX_CANDIDATES], latest


def observed_recent_growth(marks: list[dict]) -> float:
    if len(marks) < 2:
        return 0.0
    deltas = []
    for place in range(PLACES):
        if len(marks[-1]["rows"]) > place and len(marks[-2]["rows"]) > place:
            deltas.append(log_cap(marks[-1]["rows"][place]) - log_cap(marks[-2]["rows"][place]))
    return bounded((sum(deltas) / len(deltas) if deltas else 0.0) * 6.0, -0.5, 0.6)


def calibrate_candidate_coverage(marks: list[dict], cadence_minutes: int) -> list[dict[int, tuple[int, int]]]:
    """Measure how often a target-place coin was in the cut's candidate set."""
    coverage: list[dict[int, tuple[int, int]]] = [defaultdict(lambda: (0, 0)) for _ in range(PLACES)]
    for cut in range(1, len(marks) - 1):
        candidates, _ = build_candidates(marks[:cut + 1], cadence_minutes)
        known = {candidate["mint"] for candidate in candidates}
        for target in range(cut + 1, len(marks)):
            gap = target - cut
            rows = marks[target]["rows"]
            for place_index in range(min(PLACES, len(rows))):
                yes, total = coverage[place_index][gap]
                coverage[place_index][gap] = (yes + (rows[place_index]["mint"] in known), total + 1)
    return coverage


def known_mass(horizon_hours: float, place: int, coverage: list[dict[int, tuple[int, int]]], cadence_minutes: int) -> float:
    horizon = max(0.0, horizon_hours)
    target_gap = horizon * 60.0 / cadence_minutes
    prior = bounded(KNOWN_PRIOR_START - KNOWN_PRIOR_FADE * min(horizon, 13.0), 0.30, 0.58)
    bandwidth = max(1.5, min(8.0, target_gap * 0.12))
    weighted_yes = weighted_total = 0.0
    for gap, (yes, total) in coverage[place - 1].items():
        z = (gap - target_gap) / bandwidth
        weight = math.exp(-0.5 * z * z)
        weighted_yes += weight * yes
        weighted_total += weight * total
    return bounded((weighted_yes + 6.0 * prior) / (weighted_total + 6.0), 0.18, 0.72)


def simulate_horizon(
    candidates: list[dict], marks: list[dict], horizon_hours: float, place_count: int,
    edges: list[float], seed: int, coverage: list[dict[int, tuple[int, int]]],
    cadence_minutes: int,
) -> tuple[list[dict[str, float]], list[list[float]], list[float]]:
    rng = random.Random(seed)
    n = len(candidates)
    hist = [[0] * n for _ in range(place_count)]
    size_hist = [[0] * (len(edges) + 1) for _ in range(place_count)]
    horizon = max(0.04, horizon_hours)
    growth = observed_recent_growth(marks)

    for _ in range(SIMULATIONS):
        common = rng.gauss(0.0, 0.10 * math.sqrt(horizon)) + bounded(growth * min(horizon, 1.0) * 0.35, -0.25, 0.25)
        forecasts = []
        for candidate in candidates:
            trend = bounded(candidate["momentum"] * MOMENTUM_SCALE * min(horizon * 6.0, 3.0), -0.40, 0.40)
            age_drag = 0.006 * max(0.0, candidate["age"] - 2.0) * min(horizon, 8.0)
            mean = candidate["base"] + trend - age_drag - 0.025 * candidate["stale"] + common
            sigma_per_hour = 0.17 + 0.035 * min(candidate["age"], 5.0) + 0.06 * min(abs(candidate["momentum"]), 1.0)
            value = mean + rng.gauss(0.0, VOLATILITY_SCALE * sigma_per_hour * math.sqrt(horizon))
            crash_p = min(0.48, 1.0 - math.exp(-CRASH_HAZARD * horizon))
            if rng.random() < crash_p:
                value -= 0.45 + rng.expovariate(1.9)
            launch_p = min(0.34, 1.0 - math.exp(-LAUNCH_HAZARD * horizon))
            if rng.random() < launch_p:
                value += 0.25 + rng.expovariate(2.4)
            forecasts.append(value)

        order = sorted(range(n), key=lambda i: (-forecasts[i], candidates[i]["mint"]))
        for place_index in range(min(place_count, n)):
            candidate_index = order[place_index]
            hist[place_index][candidate_index] += 1
            size_hist[place_index][log_bin(forecasts[candidate_index], edges)] += 1

    probabilities = []
    for place_index in range(place_count):
        mass = known_mass(horizon, place_index + 1, coverage, cadence_minutes)
        probabilities.append({
            candidates[i]["mint"]: mass * hist[place_index][i] / SIMULATIONS
            for i in range(n) if hist[place_index][i]
        })

    known_sizes = [[count / SIMULATIONS for count in size_hist[p]] for p in range(place_count)]
    last_rows = marks[-1]["rows"]
    proxy_growth = bounded(growth * 0.18, -0.04, 0.08)
    unknown_sizes = []
    for place_index in range(place_count):
        anchor_row = last_rows[min(place_index, len(last_rows) - 1)]
        anchor = log_cap(anchor_row)
        mean = anchor + bounded(proxy_growth * min(horizon, 8.0) + UNKNOWN_SIZE_GROWTH_PER_HOUR * horizon, -0.25, 0.72)
        sd = UNKNOWN_SIZE_SD + UNKNOWN_SIZE_HORIZON_SD * math.sqrt(horizon)
        unknown_sizes.append(normal_bins(mean, sd, edges))

    mixed_sizes = []
    for place_index in range(place_count):
        mass = known_mass(horizon, place_index + 1, coverage, cadence_minutes)
        mixed_sizes.append([
            mass * known_sizes[place_index][b] + (1.0 - mass) * unknown_sizes[place_index][b]
            for b in range(len(edges) + 1)
        ])
    expected_rank = [0.0] * n
    for place_index, table in enumerate(probabilities):
        for i, candidate in enumerate(candidates):
            expected_rank[i] += (place_count - place_index) * table.get(candidate["mint"], 0.0)
    return probabilities, mixed_sizes, expected_rank


def select_probability_rows(probabilities: dict[str, float], max_rows: int = MAX_ROWS_PER_PLACE) -> list[dict]:
    ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    selected: list[tuple[str, float]] = []
    total = 0.0
    for mint, p in ranked[:max_rows]:
        residual_share = max(0.0, 1.0 - total) / max(1, 100 - len(selected))
        if p <= residual_share:
            break
        selected.append((mint, p))
        total += p
    return [{"mint": mint, "p": p} for mint, p in selected]


def forecast(task: dict, marks: list[dict]) -> dict:
    edges = task["size"]["edges"]
    places = int(task.get("places", PLACES))
    cadence_minutes = int(task.get("cadence_minutes", 10))
    candidates, _ = build_candidates(marks, cadence_minutes)
    coverage = calibrate_candidate_coverage(marks, cadence_minutes)
    if len(candidates) < places:
        raise ValueError("fewer than five candidate coins in the history")

    history_until = parse_time(task["history_until"])
    hist_bytes = "".join(row["mint"] + ":" + str(row["market_cap_lamports"]) for row in marks[-1]["rows"])
    seed_root = int.from_bytes(hashlib.sha256(hist_bytes.encode("utf-8")).digest()[:8], "big")
    probability_totals = [defaultdict(float) for _ in range(places)]
    size_totals = [[0.0] * (len(edges) + 1) for _ in range(places)]
    expected_rank = defaultdict(float)
    reads = task.get("reads") or [task["window"]["answer_at"]]

    for read_index, read in enumerate(reads):
        horizon = max(0.04, (parse_time(read) - history_until).total_seconds() / 3600.0)
        probabilities, sizes, ranks = simulate_horizon(
            candidates, marks, horizon, places, edges, seed_root + read_index * 104729,
            coverage, cadence_minutes,
        )
        for place_index in range(places):
            for mint, p in probabilities[place_index].items():
                probability_totals[place_index][mint] += p / len(reads)
            for bin_index, p in enumerate(sizes[place_index]):
                size_totals[place_index][bin_index] += p / len(reads)
        for i, candidate in enumerate(candidates):
            expected_rank[candidate["mint"]] += ranks[i] / len(reads)

    probability_rows = []
    for place_index in range(places):
        for row in select_probability_rows(dict(probability_totals[place_index])):
            probability_rows.append({"mint": row["mint"], "place": place_index + 1, "p": row["p"]})

    top5 = sorted(candidates, key=lambda c: (-expected_rank[c["mint"]], c["rank"], c["mint"]))[:places]
    top5_mints = [c["mint"] for c in top5]
    for row in marks[-1]["rows"]:
        if row["mint"] not in top5_mints and len(top5_mints) < places:
            top5_mints.append(row["mint"])

    size_rows = []
    for place_index in range(places):
        probs = [max(0.0, p) for p in size_totals[place_index]]
        total = sum(probs)
        probs = ([1.0 / len(probs)] * len(probs) if total <= 0.0
                 else [p / total for p in probs])
        probs = [round(p, 9) for p in probs]
        probs[-1] = max(0.0, round(1.0 - sum(probs[:-1]), 9))
        for bin_index, p in enumerate(probs):
            size_rows.append({"place": place_index + 1, "bin": bin_index, "p": p})

    return {"probabilities": probability_rows, "top5": top5_mints, "sizes": size_rows}


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    task, marks = load_history(input_dir)
    validate_inputs(task, marks)
    result = forecast(task, marks)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / task.get("output", "forecast.json")).write_text(
        json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )






def validate_inputs(task: dict, marks: list[dict]) -> None:
    """Reject malformed histories before they can produce a plausible-looking forecast."""
    size = task.get("size")
    edges = size.get("edges") if isinstance(size, dict) else None
    if not isinstance(edges, list) or not edges:
        raise ValueError("task.size.edges must be a non-empty list")
    if any(not isinstance(edge, (int, float)) or not math.isfinite(edge) for edge in edges):
        raise ValueError("size edges must be finite numbers")
    if edges != sorted(set(edges)):
        raise ValueError("size edges must be strictly increasing")
    if not marks:
        raise ValueError("history has no usable marks")
    previous_at = None
    for mark in marks:
        if previous_at is not None and mark["at"] <= previous_at:
            raise ValueError("history marks must be strictly chronological")
        previous_at = mark["at"]
        seen = set()
        for row in mark["rows"]:
            mint = row.get("mint")
            if not isinstance(mint, str) or not mint:
                raise ValueError("history row has no mint")
            if mint in seen:
                raise ValueError("history mark contains a duplicate mint")
            seen.add(mint)
            if int(row["market_cap_lamports"]) <= 0:
                raise ValueError("market cap must be positive")


if __name__ == "__main__":
    main()
