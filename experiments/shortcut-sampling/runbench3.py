"""Ideation-round sampler benchmark vs the LANDED in-repo sampler.  ONE scene per process.

    python runbench3.py <scene> [n_seeds] [arms_csv]   ->  <scene>-bench3.json

Protocol (per task + audit methodology fixes):
  * budget = the landed shipped rule: max_iter=1000, max_attempts=5*max_iter
  * resolution/tolerance = the SCENE'S OWN (env.collision_resolution / env.collision_tolerance),
    not the 0.01/0.01 of bench2 -- the audit flagged the 1cm tolerance as a hole
  * arms are INTERLEAVED across repetitions (seed-major loop, arm order rotated per seed)
    for wall-time hygiene, after one unrecorded warmup run
  * full per-check curves (configs / edges / cost / wall) for the matched-axis analyses
  * invariants per run: states/modes unchanged, endpoints bitwise, cost monotone, output
    states collision-free
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import benchenv as B
import ideation_samplers as IS

SCENE = sys.argv[1]
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ARMS = (sys.argv[3].split(",") if len(sys.argv) > 3
        else ["baseline", "r2", "sweep_bisect", "sweep_bisect_loc", "gainprop"])

MAX_ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
MAX_ATTEMPTS = 5 * MAX_ITER

env, raw, meta = B.load_scene(SCENE)
from multi_robot_multi_goal_planning.problems.planning_env import State

RESOLUTION = float(env.collision_resolution)
TOLERANCE = float(env.collision_tolerance)
print(f"[bench3] scene resolution={RESOLUTION} tolerance={TOLERANCE}", flush=True)

QS = [np.array(s.q.state(), dtype=float, copy=True) for s in raw]
MODES = [s.mode for s in raw]
conf = type(raw[0].q)
sl = raw[0].q._array_slice


def fresh():
    return [State(conf(np.array(q, copy=True), sl), m) for q, m in zip(QS, MODES)]


PREP = B.prepared_path(env, fresh())
PREP_METRICS = B.path_metrics(env, PREP)
PREP_MODES = [tuple(int(t) for t in np.atleast_1d(s.mode.task_ids)) for s in PREP]
PREP_Q0 = np.array(PREP[0].q.state(), copy=True)
PREP_QN = np.array(PREP[-1].q.state(), copy=True)
print(f"[bench3] prepared: {len(PREP)} states, cost {PREP_METRICS['cost']:.4f}", flush=True)

counter = B.count_collision_checks(env)

# ---- fail-capture: record the q of the last FAILING state/edge check --------------------
FAIL = {"q": None}
_inner_edge = env.is_edge_collision_free
_inner_cfg = env.is_collision_free


def _edge_capture(q1, *a, **kw):
    r = _inner_edge(q1, *a, **kw)
    if not r:
        FAIL["q"] = q1
    return r


def _cfg_capture(q, *a, **kw):
    r = _inner_cfg(q, *a, **kw)
    if not r:
        FAIL["q"] = q
    return r


env.is_edge_collision_free = _edge_capture
env.is_collision_free = _cfg_capture

# ---- recorder: (configs, edges, wall) after every candidate path check ------------------
_rec = []
_inner_pcf = env.is_path_collision_free
_T0 = [0.0]


def _recording_pcf(*a, **kw):
    r = _inner_pcf(*a, **kw)
    _rec.append((counter.configs, counter.edges, time.time()))
    return r


env.is_path_collision_free = _recording_pcf


def invariants(out, costs):
    inv = {}
    inv["n_states_unchanged"] = (len(out) == len(PREP))
    omodes = [tuple(int(t) for t in np.atleast_1d(s.mode.task_ids)) for s in out]
    inv["modes_unchanged"] = (omodes == PREP_MODES)
    inv["start_drift"] = float(np.linalg.norm(out[0].q.state() - PREP_Q0))
    inv["end_drift"] = float(np.linalg.norm(out[-1].q.state() - PREP_QN))
    inv["endpoints_within_1e9"] = bool(inv["start_drift"] < 1e-9 and inv["end_drift"] < 1e-9)
    d = np.diff(np.asarray(costs))
    inv["max_cost_increase"] = float(d.max()) if len(d) else 0.0
    inv["cost_monotone"] = bool(inv["max_cost_increase"] <= 1e-9)
    bad = B.is_path_valid(env, out)
    inv["colliding_states"] = bad
    inv["collision_free"] = (len(bad) == 0)
    inv["all_ok"] = all([inv["n_states_unchanged"], inv["modes_unchanged"],
                         inv["endpoints_within_1e9"], inv["cost_monotone"],
                         inv["collision_free"]])
    return inv


def one_run(arm, seed, max_iter=MAX_ITER, max_attempts=MAX_ATTEMPTS, record=True):
    _rec.clear()
    counter.reset()
    FAIL["q"] = None
    np.random.seed(seed)  # the baseline arm consumes global np.random like the landed code
    st = {}
    t0 = time.time()
    out, (costs, times) = IS.shortcut_run(
        env, fresh(), arm, max_iter=max_iter, max_attempts=max_attempts,
        resolution=RESOLUTION, tolerance=TOLERANCE, interpolation_resolution=0.5,
        seed=seed, fail_cell=FAIL, stats_out=st, check_counter=counter)
    wall = time.time() - t0

    n = min(len(_rec), len(costs) - 1)
    curve = {
        "pchecks": list(range(1, n + 1)),
        "configs": [int(_rec[k][0]) for k in range(n)],
        "edges": [int(_rec[k][1]) for k in range(n)],
        "cost": [float(costs[k + 1]) for k in range(n)],
        "time": [float(_rec[k][2] - t0) for k in range(n)],
    } if record else None
    inv = invariants(out, costs) if record else None
    post = B.path_metrics(env, out) if record else None
    rec = dict(
        arm=arm, seed=seed, wall=wall,
        cost_initial=float(costs[0]), cost_final=float(costs[-1]),
        configs=int(_rec[n - 1][0]) if n else 0,
        edges=int(_rec[n - 1][1]) if n else 0,
        pchecks=int(n),
        us_per_config=1e6 * wall / max(1, int(_rec[n - 1][0]) if n else 1),
        sampler_time=st.get("sampler_time"), sampler_calls=st.get("sampler_calls"),
        sampler_us_per_call=(1e6 * st.get("sampler_time", 0.0)
                             / max(1, st.get("sampler_calls", 1))),
        stop_reason=st.get("stop_reason"), iters=st.get("iters"),
        skipped_short=st.get("skipped_short"), skipped_gate=st.get("skipped_gate"),
        cost_rejects=st.get("cost_rejects"), accepted=st.get("accepted"),
        rejected_collision=st.get("rejected_collision"),
        collision_checks=st.get("collision_checks"),
        fail_located=st.get("fail_located"),
        extra={k: st[k] for k in ("sweep_total", "sweep_proposed", "sweep_accepted",
                                  "pops", "splits", "split_located", "stack_dropped",
                                  "stack_left", "g0", "recomputes", "G_left",
                                  "r2_retry_exhausted") if k in st},
    )
    if record:
        rec["detour_after"] = post["total_detour"]
        rec["length_after"] = post["total_length"]
        rec["invariants"] = inv
        rec["curve"] = curve
    return rec


# ---- warmup (unrecorded) ----------------------------------------------------------------
print("[bench3] warmup ...", flush=True)
one_run("baseline", 987654, max_iter=30, max_attempts=200, record=False)

# ---- main sweep: seed-major, arm order rotated per seed ---------------------------------
runs = []
for seed in range(NSEED):
    order = ARMS[seed % len(ARMS):] + ARMS[:seed % len(ARMS)]
    for arm in order:
        r = one_run(arm, seed)
        runs.append(r)
        flag = "" if r["invariants"]["all_ok"] else "   <<<< INVARIANT VIOLATION"
        print(f"[{arm:17s} s{seed}] {r['cost_initial']:.2f}->{r['cost_final']:.2f} "
              f"chk={r['pchecks']} cfg={r['configs']} acc={r['accepted']} "
              f"iters={r['iters']} short={r['skipped_short']} costrej={r['cost_rejects']} "
              f"sampler={1e3*r['sampler_time']:.0f}ms {r['wall']:.1f}s "
              f"({r['stop_reason']}) {r['extra']}{flag}", flush=True)

out_json = os.path.join(HERE, f"{SCENE}-bench3.json")
json.dump({"scene": SCENE,
           "meta": {k: v for k, v in meta.items()
                    if isinstance(v, (int, float, str, bool, list, type(None)))},
           "prepared": PREP_METRICS,
           "resolution": RESOLUTION, "tolerance": TOLERANCE,
           "budget": {"max_iter": MAX_ITER, "max_attempts": MAX_ATTEMPTS},
           "arms": ARMS, "n_seeds": NSEED,
           "runs": runs},
          open(out_json, "w"), indent=1, default=float)
print("wrote", out_json)
