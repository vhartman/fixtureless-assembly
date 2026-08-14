"""Paired numerical summary for the fixed-environment shortcut matrix."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = "shortcut-portfolio-v4-preserve-input-edges"
POLICIES = (
    "whole_path_random", "native", "persistent_tree", "lazy_portfolio",
    "containment_portfolio",
)
ENVS = (
    "triple", "box", "welding", "handover", "hallway", "rearrangement",
    "box_two_arm",
)


def _load(phase):
    out = {}
    pattern = HERE / f"live-persistent-v4-{phase}-*.json"
    for filename in glob.glob(str(pattern)):
        try:
            with open(filename) as stream:
                data = json.load(stream)
        except Exception:
            continue
        cert = data.get("certificate", {})
        if data.get("protocol") != PROTOCOL or not cert.get("strict_full_edge"):
            continue
        key = (data["env"], int(data["run_id"]), data["policy"])
        out[key] = data
    return out


def _step(x, y, grid):
    indices = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(indices, 0)]


def _curve(data, phase):
    if phase == "standalone":
        call = data["patch"]["shortcut_calls"][-1]
        curve = call["curve"]
        setup = float(call.get("trace_setup_offset_seconds", 0.0))
        return (
            np.concatenate((np.asarray([0.0]), setup + np.asarray(curve["time"], dtype=float))),
            np.concatenate((
                np.asarray([float(call["input"]["cost"])]),
                np.asarray(curve["cost"], dtype=float),
            )),
        )
    else:
        curve = data["full_curve"]
    return np.asarray(curve["time"], dtype=float), np.asarray(curve["cost"], dtype=float)


def _bootstrap_mean_ci(values, rng, repetitions=20000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def analyze(phase, allowed_run_ids=None):
    data = _load(phase)
    rng = np.random.default_rng(20260812)
    by_env = {}
    all_relative_auc = defaultdict(list)
    all_relative_final = defaultdict(list)
    all_auc_wins = defaultdict(int)
    all_final_wins = defaultdict(int)
    paired_total = 0

    for env in ENVS:
        registry_names = sorted({key[0] for key in data if key[0].endswith({
            "triple": "triple_waypoints", "box": "box_stacking",
            "welding": "welding", "handover": "2d_handover",
            "hallway": "other_hallway", "rearrangement": "box_rearrangement",
            "box_two_arm": "box_stacking_two_robots",
        }[env])})
        if len(registry_names) != 1:
            continue
        registry = registry_names[0]
        common_runs = set.intersection(*(
            {run_id for name, run_id, arm in data if name == registry and arm == policy}
            for policy in POLICIES
        ))
        if allowed_run_ids is not None:
            common_runs &= allowed_run_ids
        run_ids = sorted(common_runs)
        if not run_ids:
            continue

        fingerprints = {
            data[(registry, run_id, policy)]["environment_fingerprint"]
            for run_id in run_ids for policy in POLICIES
        }
        if len(fingerprints) != 1:
            raise RuntimeError(f"environment changed inside {env}: {fingerprints}")

        if phase == "standalone":
            for run_id in run_ids:
                signatures = set()
                for policy in POLICIES:
                    call = data[(registry, run_id, policy)]["patch"]["shortcut_calls"][-1]
                    signatures.add((
                        call["input"]["states"], round(call["input"]["cost"], 12),
                        tuple(tuple(mode) for mode in call["input"]["mode_runs"]),
                    ))
                if len(signatures) != 1:
                    raise RuntimeError(f"unmatched input paths for {env} run {run_id}")

        finals = defaultdict(list)
        aucs = defaultdict(list)
        checks = defaultdict(list)
        sampler_fractions = defaultdict(list)
        selection_fractions = defaultdict(list)
        first_solution_times = defaultdict(list)

        for run_id in run_ids:
            group = {policy: data[(registry, run_id, policy)] for policy in POLICIES}
            curves = {policy: _curve(group[policy], phase) for policy in POLICIES}
            start = max(float(x[0]) for x, _ in curves.values()) if phase == "full" else 0.0
            finish = max(float(x[-1]) for x, _ in curves.values())
            grid = np.linspace(start, finish, 2001)
            horizon = finish - start
            for policy, (x, y) in curves.items():
                finals[policy].append(float(y[-1]))
                values = _step(x, y, grid)
                aucs[policy].append(
                    float(np.trapezoid(values, grid) / horizon) if horizon > 0 else float(y[-1])
                )
                first_solution_times[policy].append(float(x[0]))
                calls = group[policy]["patch"]["shortcut_calls"]
                checks[policy].append(sum(
                    int(call["collision_checks"]["paths"]) for call in calls
                ))
                stats = [call.get("stats") for call in calls if call.get("stats")]
                if stats:
                    wall = max(sum(float(item["wall_time"]) for item in stats), 1e-12)
                    sampler_fractions[policy].append(
                        sum(float(item["sampler_time"]) for item in stats) / wall
                    )
                    if all("portfolio_selection_time" in item for item in stats):
                        selection_fractions[policy].append(
                            sum(float(item["portfolio_selection_time"]) for item in stats) / wall
                        )

        policies = {}
        native_final = np.asarray(finals["native"])
        native_auc = np.asarray(aucs["native"])
        for policy in POLICIES:
            final = np.asarray(finals[policy])
            auc = np.asarray(aucs[policy])
            delta_final = final - native_final
            delta_auc = auc - native_auc
            relative_final = delta_final / native_final
            relative_auc = delta_auc / native_auc
            all_relative_final[policy].extend(relative_final.tolist())
            all_relative_auc[policy].extend(relative_auc.tolist())
            all_final_wins[policy] += int(np.sum(delta_final < -1e-12))
            all_auc_wins[policy] += int(np.sum(delta_auc < -1e-12))
            policies[policy] = {
                "final_cost_mean": float(final.mean()),
                "final_cost_sd": float(final.std(ddof=1)),
                "auc_cost_mean": float(auc.mean()),
                "auc_cost_sd": float(auc.std(ddof=1)),
                "paired_final_delta_vs_native_mean": float(delta_final.mean()),
                "paired_final_delta_vs_native_ci95": _bootstrap_mean_ci(delta_final, rng),
                "paired_auc_delta_vs_native_mean": float(delta_auc.mean()),
                "paired_auc_delta_vs_native_ci95": _bootstrap_mean_ci(delta_auc, rng),
                "paired_final_wins_vs_native": int(np.sum(delta_final < -1e-12)),
                "paired_auc_wins_vs_native": int(np.sum(delta_auc < -1e-12)),
                "mean_total_shortcut_path_collision_calls": float(np.mean(checks[policy])),
                "mean_first_solution_time": float(np.mean(first_solution_times[policy])),
                "mean_sampler_wall_fraction": (
                    float(np.mean(sampler_fractions[policy]))
                    if sampler_fractions[policy] else None
                ),
                "mean_portfolio_selection_wall_fraction": (
                    float(np.mean(selection_fractions[policy]))
                    if selection_fractions[policy] else None
                ),
            }

        by_env[env] = {
            "registry_name": registry,
            "run_ids": run_ids,
            "n_paired": len(run_ids),
            "environment_fingerprint": next(iter(fingerprints)),
            "policies": policies,
        }
        paired_total += len(run_ids)

    overall = {}
    for policy in POLICIES:
        rel_final = np.asarray(all_relative_final[policy])
        rel_auc = np.asarray(all_relative_auc[policy])
        overall[policy] = {
            "paired_environment_runs": int(len(rel_auc)),
            "mean_relative_final_delta_vs_native": float(rel_final.mean()),
            "mean_relative_auc_delta_vs_native": float(rel_auc.mean()),
            "paired_final_wins_vs_native": int(all_final_wins[policy]),
            "paired_auc_wins_vs_native": int(all_auc_wins[policy]),
        }

    return {
        "protocol": PROTOCOL,
        "phase": phase,
        "auc_definition": (
            "stepwise cost integrated over each paired run's common post-solution wall-time "
            "window and divided by window length; completed policies hold final incumbent"
        ),
        "paired_environment_runs": paired_total,
        "environments": by_env,
        "overall": overall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("standalone", "full"), required=True)
    parser.add_argument("--run-ids", help="optional comma-separated matched run IDs")
    args = parser.parse_args()
    allowed = (
        {int(value) for value in args.run_ids.split(",") if value}
        if args.run_ids else None
    )
    result = analyze(args.phase, allowed)
    run_tag = f"{len(allowed)}run" if allowed is not None else "10run"
    output = ROOT / f"shortcut_portfolio_mrbench_{run_tag}_{args.phase}_summary.json"
    with output.open("w") as stream:
        json.dump(result, stream, indent=2)
    print(output)
    for env, values in result["environments"].items():
        print(env, "n=", values["n_paired"])
        for policy, stats in values["policies"].items():
            print(
                f"  {policy:25s} final={stats['final_cost_mean']:.3f} "
                f"auc={stats['auc_cost_mean']:.3f} "
                f"dAUC={stats['paired_auc_delta_vs_native_mean']:+.3f} "
                f"wins={stats['paired_auc_wins_vs_native']}/{values['n_paired']}"
            )
    print("overall")
    for policy, stats in result["overall"].items():
        print(policy, stats)


if __name__ == "__main__":
    main()
