"""Re-stamp a raw fixture's saved task signature with the stable (sorted) hash.

    python restamp_signatures.py <scene>

One scene per process.  Rebuilds the problem from the sidecar, loads the EXISTING raw path
states (no re-planning), and rewrites ``raw/<scene>-rawpath.json`` with
``benchenv.stable_task_signature``.  The old saved hashes were computed with the raw
mp_export function whose per-task list order varies across processes (set iteration +
hash randomization), so a fixture saved in one process could fail the guard in the next.

New file; no experiment data is changed, only the stored signature.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import benchenv as B
import mp_export as MP


def main():
    scene = sys.argv[1]
    meta = B.scene_meta(scene)
    prob, cfg, plan, _ = B._build_problem(meta["plan"], meta["mp_collision_tolerance"])
    raw_path = os.path.join(RAW, f"{scene}-rawpath.json")
    out, want, have = B._load_real_path(prob, MP, raw_path)
    sig = B.stable_task_signature(prob)
    with open(raw_path, "w") as f:
        json.dump({"states": [s.to_dict() for s in out], "task_signature": sig}, f, indent=2)
    MP.load_path(raw_path)
    if MP.load_path.last_signature != sig:
        raise SystemExit("roundtrip signature mismatch after re-stamp")
    print(f"[restamp] {scene}: {len(out)} states, old signature {want}, "
          f"new stable signature {sig}")


if __name__ == "__main__":
    main()
