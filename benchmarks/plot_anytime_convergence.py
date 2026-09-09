#!/usr/bin/env python3
"""Plot incumbent cost over time from milestone-anytime or ladder planner logs.

The parser deliberately records only plans the solver reports as validated incumbents or
strict improvements.  A deadline extends the last incumbent horizontally; it never fabricates
an improvement or an optimality certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


INCUMBENT_RE = re.compile(
    r"\[mile-anytime\] incumbent cost ([0-9.]+).* in ([0-9.]+)s")
IMPROVED_RE = re.compile(
    r"\[mile-anytime\] improved [0-9.]+ -> ([0-9.]+) in ([0-9.]+)s")
LNS_EVENT_RE = re.compile(
    r"\[lns\] k=[0-9]+ .* -> total ([0-9.]+) at ([0-9.]+)s")
LNS_SUMMARY_RE = re.compile(
    r"\[mile-anytime\] LNS improved [0-9.]+ -> ([0-9.]+) in ([0-9.]+)s")
RETURN_RE = re.compile(
    r"\[mile-anytime\] returning .*\(cost ([0-9.]+)\) in ([0-9.]+)s")
EXPANSIONS_RE = re.compile(r"\[PEA\*\] deadline reached after ([0-9]+) expansions")
LADDER_INCUMBENT_RE = re.compile(
    r"\[ladder\] rung k=[0-9]+: cost ([0-9.]+).* at ([0-9.]+)s")
LADDER_LNS_RE = re.compile(
    r"\[ladder\] LNS: [0-9.]+ -> ([0-9.]+) at ([0-9.]+)s")
LADDER_PROOF_IMPROVED_RE = re.compile(
    r"\[ladder\] proof pass improved to ([0-9.]+) at ([0-9.]+)s")
LADDER_OPTIMAL_RE = re.compile(
    r"\[ladder\] PROVEN OPTIMAL: (?:incumbent )?cost ([0-9.]+).*?"
    r"(?:in|\()\s*([0-9.]+)s")
LADDER_END_RE = re.compile(
    r"\[ladder\] (?:LNS: no improvement \(fixpoint\) at|proof (?:pass|rung) ended .* in) "
    r"([0-9.]+)s")


def parse_log(path: Path, label: str) -> dict:
  events = []
  final_time = None
  final_cost = None
  expansions = None
  optimal = False

  for line in path.read_text().splitlines():
    hit = INCUMBENT_RE.search(line)
    if hit:
      events.append({"time_s": float(hit.group(2)), "cost": float(hit.group(1)),
                     "kind": "incumbent"})
      continue
    hit = IMPROVED_RE.search(line)
    if hit:
      events.append({"time_s": float(hit.group(2)), "cost": float(hit.group(1)),
                     "kind": "improvement"})
      continue
    hit = LNS_EVENT_RE.search(line)
    if hit:
      events.append({"time_s": float(hit.group(2)), "cost": float(hit.group(1)),
                     "kind": "improvement"})
      continue
    hit = LNS_SUMMARY_RE.search(line)
    if hit:
      final_cost, final_time = float(hit.group(1)), float(hit.group(2))
      continue
    hit = RETURN_RE.search(line)
    if hit:
      final_cost, final_time = float(hit.group(1)), float(hit.group(2))
      continue
    hit = LADDER_INCUMBENT_RE.search(line)
    if hit:
      events.append({"time_s": float(hit.group(2)), "cost": float(hit.group(1)),
                     "kind": "incumbent" if not events else "improvement"})
      continue
    hit = LADDER_LNS_RE.search(line)
    if hit:
      cost, when = float(hit.group(1)), float(hit.group(2))
      # The ladder summary is emitted after the full sweep.  Verbose runs already emitted
      # the accepted splice at its actual discovery time via ``[lns]``; retain that time and
      # use the summary only to extend the final incumbent.  Quiet logs still get an event.
      if not events or events[-1]["cost"] != cost:
        events.append({"time_s": when, "cost": cost,
                       "kind": "improvement" if events else "incumbent"})
      final_cost, final_time = cost, when
      continue
    hit = LADDER_PROOF_IMPROVED_RE.search(line)
    if hit:
      events.append({"time_s": float(hit.group(2)), "cost": float(hit.group(1)),
                     "kind": "improvement"})
      continue
    hit = LADDER_OPTIMAL_RE.search(line)
    if hit:
      final_cost, final_time = float(hit.group(1)), float(hit.group(2))
      optimal = True
      if not events or events[-1]["cost"] != final_cost:
        events.append({"time_s": final_time, "cost": final_cost,
                       "kind": "improvement" if events else "incumbent"})
      continue
    hit = LADDER_END_RE.search(line)
    if hit:
      final_time = float(hit.group(1))
      if events:
        final_cost = events[-1]["cost"]
      continue
    hit = EXPANSIONS_RE.search(line)
    if hit:
      expansions = int(hit.group(1))
    if "proved optimal" in line.lower():
      optimal = True

  if not events:
    raise ValueError(f"{path}: no anytime incumbent event found")
  if final_time is None:
    final_time = events[-1]["time_s"]
    final_cost = events[-1]["cost"]
  return {
      "label": label,
      "log": str(path),
      "events": events,
      "final_time_s": final_time,
      "final_cost": final_cost,
      "exact_expansions": expansions,
      "optimal_proved": optimal,
  }


def draw(ax, run: dict, color=None) -> None:
  times = [event["time_s"] for event in run["events"]]
  costs = [event["cost"] for event in run["events"]]
  if run["final_time_s"] > times[-1]:
    times.append(run["final_time_s"])
    costs.append(costs[-1])
  line, = ax.step(times, costs, where="post", linewidth=2.2, color=color,
                  label=run["label"])
  ax.scatter([run["events"][0]["time_s"]], [run["events"][0]["cost"]],
             s=45, color=line.get_color(), zorder=3)
  for event in run["events"][1:]:
    ax.scatter([event["time_s"]], [event["cost"]], marker="D", s=38,
               color=line.get_color(), zorder=3)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", action="append", type=Path, required=True)
  parser.add_argument("--label", action="append", required=True)
  parser.add_argument("--output-prefix", type=Path, required=True)
  parser.add_argument("--title", default="Milestone-anytime convergence",
                      help="title for the combined plot")
  args = parser.parse_args()
  if len(args.log) != len(args.label):
    parser.error("each --log requires one corresponding --label")

  runs = [parse_log(path, label) for path, label in zip(args.log, args.label)]
  args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

  json_path = args.output_prefix.with_suffix(".json")
  csv_path = args.output_prefix.with_suffix(".csv")
  json_path.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
  with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=("scene", "time_s", "cost", "kind"))
    writer.writeheader()
    for run in runs:
      for event in run["events"]:
        writer.writerow({"scene": run["label"], **event})

  fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
  for run in runs:
    draw(ax, run)
  ax.set_xlim(left=0)
  ax.set_xlabel("Planner time (s)")
  ax.set_ylabel("Best validated action cost")
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  ax.grid(True, alpha=0.28)
  ax.legend(frameon=False)
  ax.set_title(args.title)
  fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
  fig.savefig(args.output_prefix.with_suffix(".pdf"))
  plt.close(fig)

  for run in runs:
    slug = re.sub(r"[^a-z0-9]+", "-", run["label"].lower()).strip("-")
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    draw(ax, run, color="#3568a8")
    ax.set_xlim(0, run["final_time_s"] * 1.02)
    ax.set_xlabel("Planner time (s)")
    ax.set_ylabel("Best validated action cost")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.28)
    status = "optimal" if run["optimal_proved"] else "optimality not proved"
    ax.set_title(f"{run['label']}: {status}")
    ax.legend(frameon=False)
    fig.savefig(args.output_prefix.parent / f"{slug}-convergence.png", dpi=180)
    fig.savefig(args.output_prefix.parent / f"{slug}-convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
  main()
