#!/usr/bin/env python3
"""Sequential, resource-bounded Fabrica geometry/planning/execution smoke tests.

Uses the named standard profiles, fresh plans and full part counts.  Search is evaluated over
a fixed 64-grasp alphabet: neither grasp-prefix widening nor action-family widening is used,
so every model exercises the same complete local-fill algorithm. Diagnostic reductions
(``--no-wrench``, ``--witness-sets 1``, ``--skip-witness-optimization``) must be requested
explicitly. No collision exemptions
or fallback to older plans/recordings are applied.
Each stage has a separate log; failure stops that model, not the remaining models.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"beam": 5, "car": 5, "duct": 8, "cooling-manifold": 7,
          "gamepad": 6, "plumbers-block": 5, "circular-stool": 9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--stages", nargs="+", choices=["audit", "search", "mp", "sim"],
                        default=["audit", "search", "mp", "sim"])
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/fabrica-validated")
    parser.add_argument("--seconds", type=int, default=600, help="wall-clock cap per stage")
    parser.add_argument("--mp-runtime", type=float, default=None, metavar="S",
                        help="override the motion profile's internal optimization runtime; "
                             "useful for staged floating-EE diagnostics")
    parser.add_argument("--memory-gb", type=float, default=4, help="address-space cap per child")
    parser.add_argument(
        "--scatter-spacing", type=float, default=0.18, metavar="M",
        help="centre-to-centre pitch of the loose-part grid (default 0.18m). The imported "
             "CAD used 1/7m historically, which leaves no open-palm approach to a circular-"
             "stool leg; this changes only the initial layout, not assembly geometry.")
    wrench = parser.add_mutually_exclusive_group()
    wrench.add_argument("--wrench", dest="wrench", action="store_true", default=True,
                        help="retain the standard profile's wrench checks (default)")
    wrench.add_argument("--no-wrench", dest="wrench", action="store_false",
                        help="explicit geometric diagnostic ablation")
    parser.add_argument(
        "--insert-force", type=float, default=0.1, metavar="N",
        help="declared Fabrica process load (default 0.1N). The source assets contain no "
             "force metadata; inheriting the generic 10N default accidentally turns this "
             "geometry benchmark into a bracing stress test.")
    parser.add_argument(
        "--wrench-profile", choices=["max_robotiq"], default="max_robotiq", metavar="NAME",
        help="persisted gripper capability profile (default max_robotiq, the hardware "
             "ceiling rather than the pending measured slip calibration)")
    parser.add_argument("--witness-sets", type=int, default=None, metavar="K",
                        help="override the standard four IK branches (e.g. 1 for a diagnostic)")
    parser.add_argument("--skip-witness-optimization", action="store_true",
                        help="skip the post-search witness-floor optimization while retaining "
                             "the exact witnesses certified by lazy search")
    parser.add_argument("--mate-robustness", type=float, default=.001, metavar="M",
                        help="audit seated parts under +/-M translations before search "
                             "(default 1mm, matching the witness mate-residual gate; 0 disables)")
    parser.add_argument("--floating-ee", action="store_true",
                        help="run search/motion planning with free 6-DoF end effectors before "
                             "testing the hardware robot; simulation is intentionally unsupported")
    parser.add_argument(
        "--sim-drive", choices=("actuated", "kinematic"), default="actuated",
        help="robot dynamics used by the simulation stage (default actuated). 'kinematic' "
             "still runs MuJoCo parts, contacts, grasp/mate welds and the saved real-arm path, "
             "but writes the certified arm configuration exactly; it is the execution-layer "
             "isolation used when diagnosing an actuator/controller failure.")
    parser.add_argument("--num-robots", type=int, default=None,
                        help="override the historical scene robot count")
    parser.add_argument("--robot", default=None,
                        help="override the historical robot model (for example ur5_limited)")
    parser.add_argument("--grippers", default=None,
                        help="comma-separated grippers; count must match --num-robots when both "
                             "are supplied")
    args = parser.parse_args()
    if (args.seconds <= 0 or args.memory_gb <= 0 or args.scatter_spacing <= 0
            or (args.mp_runtime is not None and args.mp_runtime <= 0)
            or args.mate_robustness < 0 or args.insert_force < 0
            or (args.witness_sets is not None and args.witness_sets < 1)):
        parser.error("resource limits must be positive")
    if args.num_robots is not None and args.num_robots < 1:
        parser.error("--num-robots must be positive")
    if args.grippers is not None:
        grippers = [x.strip() for x in args.grippers.split(",") if x.strip()]
        if not grippers:
            parser.error("--grippers must contain at least one gripper")
        if args.num_robots is not None and len(grippers) != args.num_robots:
            parser.error("--grippers count must match --num-robots")
    if args.floating_ee and "sim" in args.stages:
        parser.error("--floating-ee supports audit/search/mp, not simulation")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1",
               NUMBA_CACHE_DIR="/tmp/assembly-numba-cache",
               MPLCONFIGDIR="/tmp/assembly-mpl-cache")
    limit = int(args.memory_gb * 1024**3)

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    results = []
    for name in args.models:
        plan, mp = out / f"{name}-plan.json", out / f"{name}-mp.json"
        commands = {
            "audit": ["tools/diagnostics/audit_fabrica.py", name, "--containment",
                      "--tolerance", "0.0001",
                      "--robustness", str(args.mate_robustness)],
            # The historical JSON supplies only scene identity/robot layout. Reapply the
            # current named profile explicitly: otherwise every old resolved search field in
            # that JSON silently wins and this runner does not test today's standard at all.
            "search": ["main.py", "--config", f"experiments/fabrica-e2e/{name}-plan.json",
                       "--search-profile", "standard", "--solve", "--parts", str(MODELS[name]),
                       "--scatter-spacing", str(args.scatter_spacing),
                       "--insert-force", str(args.insert_force),
                       # Keep grasp generation outside the search comparison.  K=64 contains
                       # the late valid Plumbers Block contact (g55), while a single complete
                       # local graph avoids the model-dependent action-widening schedule.
                       "--grasp-sampler-keep", "64", "--grasp-sampler-candidates", "96",
                       "--no-grasp-widen", "--no-fill-action-widening",
                       # These imported CAD models contain fitted mates.  A soft KOMO
                       # objective can be individually collision-free yet freeze a slightly
                       # off-pose attachment which overlaps after later mates are accumulated.
                       # Make the interface contract explicit instead of hoping that a
                       # weight happens to seat every part accurately enough.
                       "--mate-eq-weight", "10000", "--mate-accuracy", "0.00004",
                       "--mate-angular-accuracy", "0.05",
                       "--grasp-eq-weight", "100", "--grasp-accuracy", "0.00004",
                       "--place-eq-weight", "10000",
                       "--quiet-rai", "--no-view", "--save", str(plan)],
            "mp": ["main.py", "--config", str(plan), "--load-plan", str(plan),
                   "--motion-profile", "standard", "--motion-plan",
                   "--quiet-rai", "--no-view",
                   "--mp-save", str(mp)],
            "sim": ["sim.py", "run", str(plan), "--mp-path", str(mp),
                    "--record",
                    str(out / f"{name}-standard.npz"), "--require-success", "--report-json",
                    str(out / f"{name}-sim-report.json")],
        }
        if not args.wrench:
            commands["search"].append("--no-wrench")
        if args.wrench_profile is not None:
            commands["search"] += ["--wrench-profile", args.wrench_profile]
        if args.num_robots is not None:
            commands["search"] += ["--num-robots", str(args.num_robots)]
        if args.robot is not None:
            commands["search"] += ["--robot", args.robot]
        if args.grippers is not None:
            commands["search"] += ["--grippers", args.grippers]
        if args.witness_sets is not None:
            commands["search"] += ["--witness-sets", str(args.witness_sets)]
        if args.skip_witness_optimization:
            commands["search"] += ["--optimize-witnesses", "off"]
        if args.floating_ee:
            commands["search"].append("--floating-ee")
        if args.mp_runtime is not None:
            commands["mp"] += ["--mp-runtime", str(args.mp_runtime)]
        if args.sim_drive == "kinematic":
            # This is an execution-layer isolation, not a retiming benchmark.  Follow the
            # already certified multi-robot path in lockstep and avoid spending minutes
            # recomputing a coordination-space schedule whose actuator tracking is disabled.
            commands["sim"] += ["--no-actuated", "--articulated", "--no-spline",
                                "--no-phase-schedule", "--no-phase-smooth",
                                "--no-phase-batch", "--no-phase-intent"]
        for stage in args.stages:
            command = [sys.executable, "-u", *commands[stage]]
            log = out / f"{name}-{stage}.log"
            start = time.monotonic()
            started_ns = time.time_ns()
            print(f"[{name}/{stage}] starting; {args.seconds}s, {args.memory_gb:g} GiB cap", flush=True)
            with log.open("w") as stream:
                stream.write(json.dumps({"command": command}) + "\n")
                stream.flush()
                try:
                    code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, preexec_fn=cap,
                                          timeout=args.seconds).returncode
                except subprocess.TimeoutExpired:
                    code = 124
                    stream.write("\nFAILURE: stage wall-clock budget exceeded.\n")
                artifact = {"search": plan, "mp": mp,
                            "sim": out / f"{name}-standard.npz"}.get(stage)
                if code == 0 and artifact is not None and (
                        not artifact.exists() or artifact.stat().st_mtime_ns < started_ns):
                    code = 2
                    stream.write(f"\nFAILURE: no fresh {artifact.name}; old artifacts are not reused.\n")
            row = dict(model=name, stage=stage, returncode=code,
                       seconds=round(time.monotonic() - start, 2), log=str(log), command=command)
            results.append(row)
            # One report per invocation: preserve reports from earlier subset runs.
            report = out / ("report-" + "-".join(args.models) + "-" + "-".join(args.stages) + ".json")
            report.write_text(json.dumps(results, indent=2) + "\n")
            print(f"[{name}/{stage}] {'PASS' if code == 0 else 'FAIL'} "
                  f"exit={code}, {row['seconds']}s; {log}", flush=True)
            if code:
                break
    return int(any(row["returncode"] for row in results))


if __name__ == "__main__":
    sys.exit(main())
