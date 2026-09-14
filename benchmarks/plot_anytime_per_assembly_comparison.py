#!/usr/bin/env python3
"""Plot per-assembly anytime convergence with search-comparison solver points.

No aggregation is performed across assemblies. Each subplot contains the validated incumbent
curve from the 300 s anytime run and the initial solution returned by each successful comparison
solver. An absent solver marker means that run produced no plan; no artificial cost is assigned.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/assembly-mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmarks.run_paper_search_comparison import CASES, CASE_LABEL, METHODS


PAPER_STYLE = ROOT / "paper_2.mplstyle"
METHOD_SHORT = {
    "optimistic": "Optimistic",
    "ours": "Ours",
    "flat-lazy": "Flat lazy",
    "milestones-counting": "Milestones + counting",
    "ours-exact-fills": "Exact fills",
    "ours-fine-projection": "Fine projection",
}
MARKERS = ("o", "*", "X", "s", "^", "D")


def _case_from_anytime(run):
  """Recover the stable benchmark case id from the recorded run log path."""
  run_dir = Path(run["log"]).parent.name
  case = run_dir.split("__", 1)[0]
  if case not in CASES:
    raise ValueError(f"cannot map anytime run {run.get('label')!r} from {run['log']}")
  return case


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--anytime", type=Path, required=True,
                      help="all-scenes JSON from plot_anytime_convergence.py")
  parser.add_argument("--comparison-results", type=Path, required=True,
                      help="complete paper-comparison results.json")
  parser.add_argument("--output-prefix", type=Path, required=True)
  parser.add_argument("--paper", action="store_true")
  args = parser.parse_args(argv)

  anytime_runs = json.loads(args.anytime.read_text(encoding="utf-8"))["runs"]
  anytime = {_case_from_anytime(run): run for run in anytime_runs}
  if set(anytime) != set(CASES):
    raise ValueError(f"anytime cases differ from paper cases: {sorted(set(CASES) ^ set(anytime))}")
  comparison = json.loads(args.comparison_results.read_text(encoding="utf-8"))
  records = {(row["case"], row["variant"]): row for row in comparison
             if row.get("case") in CASES and row.get("variant") in METHODS}
  missing_records = [(case, method) for case in CASES for method in METHODS
                     if (case, method) not in records]
  if missing_records:
    raise ValueError(f"comparison matrix is incomplete: {missing_records}")

  horizon = max(float(run["final_time_s"]) for run in anytime.values())
  args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
  export = {"horizon_s": horizon, "cases": []}
  csv_rows = []
  for case in CASES:
    run = anytime[case]
    solver_rows = []
    for method in METHODS:
      record = records[(case, method)]
      point = {
          "solver": method,
          "status": record.get("status"),
          "time_s": record.get("search_pipeline_s") if record.get("success") else None,
          "cost": record.get("plan_cost") if record.get("success") else None,
      }
      solver_rows.append(point)
      csv_rows.append({"assembly": case, "kind": "solver", **point})
    export["cases"].append({"assembly": case, "anytime": run, "solvers": solver_rows})
    for event in run["events"]:
      csv_rows.append({"assembly": case, "kind": event["kind"], "solver": "ours-anytime",
                       "status": "success", "time_s": event["time_s"],
                       "cost": event["cost"]})
  args.output_prefix.with_suffix(".json").write_text(
      json.dumps(export, indent=2) + "\n", encoding="utf-8")
  with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=(
        "assembly", "kind", "solver", "status", "time_s", "cost"))
    writer.writeheader()
    writer.writerows(csv_rows)

  context = (plt.style.context(str(PAPER_STYLE))
             if args.paper and PAPER_STYLE.is_file() else plt.style.context("default"))
  with context:
    fig, axes = plt.subplots(2, 5, figsize=(16.0, 6.3), sharex=True)
    # paper_2 enables figure.autolayout, which cannot account for both a shared xlabel and the
    # failure note below it. This panel has an explicit, deterministic layout instead.
    fig.set_layout_engine(None)
    for ax, case in zip(axes.flat, CASES):
      run = anytime[case]
      times = [float(event["time_s"]) for event in run["events"]]
      costs = [float(event["cost"]) for event in run["events"]]
      # A plan remains available after proof, deadline, or resource-limited early return.
      times.append(horizon)
      costs.append(costs[-1])
      ax.step(times, costs, where="post", color="#111111", linewidth=2.3, zorder=2)
      for index, method in enumerate(METHODS):
        record = records[(case, method)]
        if not record.get("success"):
          continue
        ax.scatter(float(record["search_pipeline_s"]), float(record["plan_cost"]),
                   marker=MARKERS[index], s=62, color=f"C{index}", linewidth=1.5, zorder=3)
      ax.set_title(CASE_LABEL[case], fontsize=17)
      ax.tick_params(labelsize=13)
      ax.grid(True, alpha=0.25, linewidth=0.7)
      ax.set_xlim(0, horizon * 1.015)
    fig.supxlabel(r"Search time [s]", fontsize=21, y=0.055)
    fig.supylabel(r"Plan cost", fontsize=21, x=0.012)
    handles = [Line2D([0], [0], color="#111111", linewidth=2.3,
                      label="Ours anytime")]
    handles.extend(Line2D([0], [0], linestyle="none", marker=MARKERS[index],
                          color=f"C{index}", markersize=8, label=METHOD_SHORT[method])
                   for index, method in enumerate(METHODS))
    fig.legend(handles=handles, frameon=False, ncol=4, fontsize=13,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.4, handletextpad=0.6)
    fig.text(0.5, 0.012, "Absent marker: no solution returned within the run/resource limit.",
             ha="center", va="bottom", fontsize=11)
    fig.subplots_adjust(top=0.79, bottom=0.17, left=0.07, right=0.995,
                        hspace=0.34, wspace=0.28)
    for suffix in ("pdf", "png"):
      fig.savefig(args.output_prefix.with_suffix(f".{suffix}"),
                  dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

  print(f"[anytime-per-assembly] plot -> {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
  main()
