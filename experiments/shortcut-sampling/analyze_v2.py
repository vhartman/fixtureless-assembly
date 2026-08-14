"""Round-3 summary tables from the v5 fixed-environment matrix JSONs.

For every environment and policy this reports the final cost, the shortcut wall time, the
collision configuration count, the accepted-edit count, and the integrated anytime cost over a
common time grid (lower is better).  Relative AUC is the integrated cost divided by the native
baseline on the same grid.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL = "shortcut-portfolio-v5-exact-gate"
POLICIES = (
    "native", "persistent_tree", "tree_v2", "tree_adaptive", "tree_partial",
    "tree_warm", "tree_credit", "length_uniform", "length_tilted",
    "max_compatible", "max_bandit", "lazy_certify",
)
ENV_KEYS = ("triple", "box", "welding", "handover", "hallway", "rearrangement", "box_two_arm")


def _final_call_curve(data):
    call = data["patch"]["shortcut_calls"][-1]
    setup = float(call.get("trace_setup_offset_seconds", 0.0))
    t = np.concatenate((
        np.asarray([0.0]), setup + np.asarray(call["curve"]["time"], dtype=float),
    ))
    c = np.concatenate((
        np.asarray([float(call["input"]["cost"])]),
        np.asarray(call["curve"]["cost"], dtype=float),
    ))
    return t, c


def _integrated_cost(t, c, grid):
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.maximum(idx, 0)
    return float(c[idx].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("standalone", "full"), default="standalone")
    parser.add_argument("--pattern", default=f"live-persistent-v5-{{phase}}-*.json")
    args = parser.parse_args()

    files = sorted(glob.glob(str(HERE / args.pattern.format(phase=args.phase))))
    by_env: dict[str, dict[str, list[dict]]] = {}
    for filename in files:
        try:
            with open(filename) as stream:
                data = json.load(stream)
        except Exception:
            continue
        cert = data.get("certificate", {})
        if (
            data.get("protocol") != PROTOCOL
            or not cert.get("strict_full_edge")
            or not data.get("planner", {}).get("returned_path")
        ):
            continue
        if not data["patch"]["shortcut_calls"]:
            continue
        key = data["env"].removeprefix("rai.")
        mapping = {
            "triple_waypoints": "triple", "box_stacking": "box", "welding": "welding",
            "2d_handover": "handover", "other_hallway": "hallway",
            "box_rearrangement": "rearrangement",
            "box_stacking_two_robots": "box_two_arm",
        }
        env_key = mapping.get(key)
        if env_key not in ENV_KEYS:
            continue
        call = data["patch"]["shortcut_calls"][-1]
        stats = call.get("stats") or {}
        by_env.setdefault(env_key, {}).setdefault(data["policy"], []).append({
            "cost": float(data["output"]["cost"]),
            "wall": float(call["shortcut_wall_seconds"]),
            "configs": int(call["collision_checks"]["configs"]),
            "accepted": int(stats.get("accepted", 0)),
            "provisional": int(stats.get("provisional_accepted", 0)),
            "gate_eps": int(stats.get("gate_eps_fallback", 0)),
            "gate_neg": int(stats.get("gate_negative", 0)),
            "gate_pos": int(stats.get("gate_positive", 0)),
            "gate_ok": bool(stats.get("gate_table_ok", False)),
            "adaptive_hit": bool(stats.get("adaptive_stop_hit", False)),
            "checks_cap_hit": bool(stats.get("checks_cap_hit", False)),
            "gap": stats.get("irreplaceability_gap"),
            "curve": _final_call_curve(data),
            "stop": stats.get("stop_reason"),
            "degraded_waves": int(stats.get("degraded_waves", 0)),
            "wave_commits": int(stats.get("wave_commits", 0)),
            "wave_failures": int(stats.get("wave_failures", 0)),
            "unattributable": int(stats.get("unattributable_failures", 0)),
        })

    print(f"phase={args.phase}  files={len(files)}")
    for env_key in ENV_KEYS:
        env_data = by_env.get(env_key)
        if not env_data:
            continue
        rows = {}
        for policy in POLICIES:
            runs = env_data.get(policy)
            if not runs:
                continue
            rows[policy] = runs
        native = rows.get("native")
        if not native:
            continue
        native_t, _ = native[0]["curve"]
        grid = np.linspace(0.0, float(native_t[-1]), 200)
        print(f"\n== {env_key} ({len(rows.get('native', []))} runs) ==")
        header = (
            f"{'policy':<18}{'cost':>9}{'wall_s':>9}{'cfg_k':>9}{'acc':>6}"
            f"{'auc':>9}{'rel':>7}  notes"
        )
        print(header)
        for policy in POLICIES:
            runs = rows.get(policy)
            if not runs:
                continue
            costs = np.array([r["cost"] for r in runs])
            walls = np.array([r["wall"] for r in runs])
            configs = np.array([r["configs"] for r in runs])
            accepted = np.array([r["accepted"] for r in runs])
            aucs = np.array([
                _integrated_cost(*r["curve"], grid) for r in runs
            ])
            native_aucs = np.array([
                _integrated_cost(*r["curve"], grid) for r in native
            ])
            rel = float(aucs.mean() / max(native_aucs.mean(), 1e-12))
            notes = []
            if any(r["gate_ok"] for r in runs):
                notes.append("gate-ok")
            if sum(r["gate_eps"] for r in runs):
                notes.append(f"eps={sum(r['gate_eps'] for r in runs)}")
            if any(r["adaptive_hit"] for r in runs):
                notes.append("adpt-stop")
            if any(r["checks_cap_hit"] for r in runs):
                notes.append("cap-hit")
            if any(r["degraded_waves"] for r in runs):
                notes.append(f"degrd={sum(r['degraded_waves'] for r in runs)}")
            if any(r["wave_failures"] for r in runs):
                notes.append(f"wf={sum(r['wave_failures'] for r in runs)}")
            if any(r["unattributable"] for r in runs):
                notes.append(f"unattr={sum(r['unattributable'] for r in runs)}")
            gaps = [r["gap"] for r in runs if r["gap"] is not None]
            if gaps:
                notes.append(f"gap={np.mean(gaps):.3f}")
            print(
                f"{policy:<18}{costs.mean():>9.2f}{walls.mean():>9.2f}"
                f"{configs.mean()/1000:>9.0f}{accepted.mean():>6.0f}"
                f"{aucs.mean():>9.2f}{rel:>7.3f}  {' '.join(notes)}"
            )


if __name__ == "__main__":
    main()
