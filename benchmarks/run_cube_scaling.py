#!/usr/bin/env python3
"""Run the guarded cube 1--12 x robot-count 2--4 suite and make its plot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import plot_cube_scaling  # noqa: E402
from benchmarks import search_runner  # noqa: E402


MANIFEST = ROOT / "benchmarks" / "manifests" / "cube_scaling.json"
PIECEWISE_MANIFEST = ROOT / "benchmarks" / "manifests" / "cube_scaling_piecewise.json"


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=None,
                      help="suite directory (default: timestamped benchmark result)")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--only", default=None, metavar="REGEX",
                      help="run only expanded ids matching REGEX")
  parser.add_argument("--max-runs", type=int, default=None)
  parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
  parser.add_argument("--memory-gb", type=float, default=None, metavar="GB",
                      help="override the manifest's address-space cap per run")
  parser.add_argument("--seeds", type=int, nargs="+", default=None, metavar="S",
                      help="repeat each point for these seeds; plot their mean and population sd")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--log-y", action="store_true")
  parser.add_argument("--piecewise", action="store_true",
                      help="run the strict piece-by-piece sequence control instead of plates")
  parser.add_argument("--no-wrench", action="store_true",
                      help="override the standard profile's wrench model for this ablation")
  parser.add_argument("--fill-target-support-quotient", action="store_true",
                      help="quotient target-component holder identities in the no-wrench "
                           "local reverse PDB")
  parser.add_argument(
      "--fill-scope", default="standard",
      choices=("standard", "eager-all", "local-only"),
      help="successor scope for each local fill: standard tries target/current-component "
           "actions then widens to all loose parts; eager-all starts broad; local-only "
           "disables the completeness widening",
  )
  parser.add_argument("--lazy-edge-selector", default=None,
                      choices=("forward", "terminal-first", "reverse", "fail-first",
                               "defer-failed"),
                      help="override the profile's candidate-path IK scheduling policy")
  parser.add_argument("--component-grasp-keep", type=int, default=0, metavar="N",
                      help="bound each multi-part component to N grasp rows, with the "
                           "solver's automatic full-union completion retry")
  args = parser.parse_args(argv)

  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  label = "cube-piecewise-scaling" if args.piecewise else "cube-scaling"
  if args.no_wrench:
    label += "-no-wrench"
  if args.fill_target_support_quotient:
    label += "-target-support-quotient"
  if args.fill_scope != "standard":
    label += f"-scope-{args.fill_scope}"
  if args.component_grasp_keep:
    label += f"-component-grasps-{args.component_grasp_keep}"
  output = (args.output or ROOT / "benchmarks" / "results"
            / f"{label}-1-to-12-{stamp}").resolve()
  source_manifest = PIECEWISE_MANIFEST if args.piecewise else MANIFEST
  manifest_path = source_manifest
  if args.fill_target_support_quotient and not args.no_wrench:
    parser.error("--fill-target-support-quotient requires --no-wrench; the quotient "
                 "does not represent wrench or geometric facts")
  if args.memory_gb is not None and args.memory_gb <= 0:
    parser.error("--memory-gb must be positive")
  if (args.seeds is not None or args.no_wrench or args.lazy_edge_selector is not None
      or args.fill_target_support_quotient or args.fill_scope != "standard"
      or args.component_grasp_keep or args.memory_gb is not None):
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
      parser.error("--seeds must not contain duplicates")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if args.seeds is not None:
      manifest["seeds"] = args.seeds
    if args.memory_gb is not None:
      manifest["max_address_space_gb"] = args.memory_gb
    if args.no_wrench:
      manifest["name"] += "-no-wrench"
      manifest["description"] += " Wrench feasibility is explicitly disabled."
      manifest["defaults"].append("--no-wrench")
    if args.fill_target_support_quotient:
      manifest["name"] += "-target-support-quotient"
      manifest["description"] += (
          " Target-component holder identities are quotiented in each no-wrench local PDB.")
      manifest["defaults"].append("--fill-target-support-quotient")
    if args.fill_scope == "eager-all":
      manifest["name"] += "-scope-eager-all"
      manifest["description"] += (
          " Every local fill starts with all loose parts manipulable (scope ablation).")
      manifest["defaults"] += ["--no-fill-scope-widening", "--fill-touch-all"]
    elif args.fill_scope == "local-only":
      manifest["name"] += "-scope-local-only"
      manifest["description"] += (
          " Local fills never widen to manipulation of future loose parts (scope ablation).")
      manifest["defaults"] += ["--no-fill-scope-widening", "--no-fill-touch-all"]
    if args.lazy_edge_selector is not None:
      manifest["name"] += f"-{args.lazy_edge_selector}"
      manifest["description"] += (
          f" Lazy IK edge selector is explicitly {args.lazy_edge_selector}.")
      manifest["defaults"] += ["--lazy-edge-selector", args.lazy_edge_selector]
    if args.component_grasp_keep:
      if args.component_grasp_keep < 1:
        parser.error("--component-grasp-keep must be >= 1 when supplied")
      manifest["name"] += f"-component-grasps-{args.component_grasp_keep}"
      manifest["description"] += (
          f" Multi-part components initially expose at most {args.component_grasp_keep} "
          "whole-component-clear grasp rows, with a complete full-union retry on exhaustion.")
      manifest["defaults"] += ["--component-grasp-keep",
                                str(args.component_grasp_keep)]
    manifest_path = output / "requested-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  runner_args = [str(manifest_path), "--output", str(output)]
  if args.resume:
    runner_args.append("--resume")
  if args.only is not None:
    runner_args += ["--only", args.only]
  if args.max_runs is not None:
    runner_args += ["--max-runs", str(args.max_runs)]
  if args.timeout is not None:
    runner_args += ["--timeout", str(args.timeout)]
  if args.dry_run:
    runner_args.append("--dry-run")

  status = search_runner.main(runner_args)
  if not args.dry_run:
    # Plot partial suites as well: failures remain explicit censored markers and the non-zero
    # runner status is preserved for automation.
    plot_cube_scaling.plot_suite(output, log_y=args.log_y)
    print(f"[cube-scaling] plot -> {output / 'cube-scaling.pdf'}")
  return status


if __name__ == "__main__":
  raise SystemExit(main())
