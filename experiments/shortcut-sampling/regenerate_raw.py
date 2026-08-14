"""Regenerate a raw assembly-scene fixture from its CURRENT source plan.

    python regenerate_raw.py <scene>      # e.g. car-mini-lim

The old car-mini raw paths were planned against an older plan order; their saved task
signature no longer matches the rebuild of the current plan file.  This script (a) copies the
current source plan into ``raw/``, (b) rebuilds the problem with benchenv's own code path,
(c) plans a fresh unshortcut first-solution BiRRT path with the sidecar's recorded settings
(optimize=False, shortcut=False, runtime, seed), (d) checks the path with the strict
in-order edge certificate, and (e) rewrites ``raw/<scene>-rawpath.json`` (with task
signature) and ``raw/<scene>-meta.json``.  ONE scene per process.

New file; the original fixture generator and every experiment file are untouched.
"""
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024 ** 3, 12 * 1024 ** 3))
except (ImportError, ValueError, OSError):
    pass

import benchenv as B
import mp_export as MP


def main():
    scene = sys.argv[1]
    source = os.path.join(REPO, f"{scene}.json")
    if not os.path.exists(source):
        raise SystemExit(f"source plan not found: {source}")

    old_meta = B.scene_meta(scene)
    tolerance = float(old_meta["mp_collision_tolerance"])
    runtime = float(old_meta.get("mp_runtime", 60.0))
    seed = int(old_meta.get("mp_seed", 1))

    plan_copy = os.path.join(RAW, f"{scene}-plan.json")
    raw_copy = os.path.join(RAW, f"{scene}-rawpath.json")
    shutil.copy(source, plan_copy)
    print(f"[regen] plan copy: {source} -> {plan_copy}")

    prob, cfg, plan, _MP = B._build_problem(plan_copy, tolerance)
    print(f"[regen] problem rebuilt: {len(prob.tasks)} tasks, {len(prob.robots)} robots, "
          f"tolerance {tolerance}, signature {B.stable_task_signature(prob)}")

    t0 = time.time()
    path, info = MP.run_birrt(prob, runtime=runtime, shortcut=False, optimize=False,
                              seed=seed)
    print(f"[regen] BiRRT wall {time.time() - t0:.1f}s, states {len(path) if path else 0}")

    if not path:
        raise SystemExit("BiRRT returned no path; fixture NOT rewritten")

    ok = prob.is_path_collision_free(path, check_edges_in_order=True)
    if not ok:
        raise SystemExit("strict in-order edge certificate FAILED; fixture NOT rewritten")
    print("[regen] strict in-order edge certificate PASSED")

    sig = B.stable_task_signature(prob)
    with open(raw_copy, "w") as f:
        json.dump({"states": [s.to_dict() for s in path], "task_signature": sig}, f, indent=2)
    MP.load_path(raw_copy)
    if MP.load_path.last_signature != sig:
        raise SystemExit("roundtrip signature mismatch after save")

    meta = dict(old_meta)
    meta.update(
        plan=plan_copy,
        raw_path=raw_copy,
        regenerated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        task_signature=sig,
        note=(old_meta.get("note", "") +
              " REGENERATED: current source plan copied into raw/, fresh unshortcut "
              "first-solution path planned on the rebuild, strict edge certificate passed."),
    )
    with open(os.path.join(RAW, f"{scene}-meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[regen] wrote {raw_copy} with signature {sig}")
    print(f"[regen] wrote meta sidecar; fixture ready")


if __name__ == "__main__":
    main()
