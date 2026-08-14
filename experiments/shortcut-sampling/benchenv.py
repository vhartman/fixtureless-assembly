"""benchenv.py -- one-call loader for the RAW-path shortcutting benchmark.

    from benchenv import load_scene, scene_names
    env, raw, meta = load_scene("car-mini-lim")

`env` is the REAL rai/mr_bench problem (live collision checker, margin wrappers installed
exactly as the motion planner had them), `raw` is the UNSHORTCUT first-solution BiRRT path as
real mr_bench `State`s (real `Mode` objects taken from the rebuilt mode chain), `meta` is a
dict of everything the benchmark needs to report.

Nothing here is hard-coded per scene: every knob is read back from
  * `<name>-plan.json`  -- the plan + scene config + the persisted IK witnesses, written by
    `main.py --save` in the SAME process that wrote the path, and
  * `<name>-meta.json`  -- the sidecar with the export-time settings that `--save` does NOT
    persist (`--mp-collision-tolerance`, `--mp-seed`, the fuse flags).
`collision_resolution` is not a config at all: it is the mr_bench env default, so it is read
off the rebuilt env (`env.collision_resolution`) and reported.

HARD CONSTRAINT: build ONE scene per process. assembly/* keeps rai scene state in module
globals (wrench.PARAMS, grasp caches, ...) which leak between builds -- see the caller repo's
FRAGILITY.md. `load_scene` refuses a second build in the same interpreter.

mr_bench itself is never modified; `count_collision_checks` monkeypatches the *instance*.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = "/home/valentin/git/postdoc/robotic-venv/assembly"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_BUILT = []


# ---------------------------------------------------------------------------
# scene registry (discovered from the sidecars on disk -- no per-scene constants here)
# ---------------------------------------------------------------------------
def scene_names():
    return sorted(os.path.basename(p)[:-len("-meta.json")]
                  for p in glob.glob(os.path.join(RAW, "*-meta.json")))


def scene_meta(name):
    with open(os.path.join(RAW, f"{name}-meta.json")) as f:
        return json.load(f)


def stable_task_signature(problem):
    """Same fingerprint as mp_export.task_signature but ORDER-STABLE within each task.

    make_problem builds some per-task robot/frame lists from Python sets, so the raw
    mp_export hash varies across processes (str hash randomization) and a fixture saved in
    one process can fail to match in the next.  Sorting each per-task list before hashing
    makes the fingerprint a function of the task STRUCTURE only, which is what the guard
    checks.
    """
    import hashlib
    h = hashlib.sha256()
    for t in problem.tasks:
        robots = sorted(str(r) for r in (getattr(t, "robots", None) or []))
        frames = sorted(str(f) for f in (getattr(t, "frames", None) or []))
        h.update(("|".join([
            str(getattr(t, "name", "")),
            str(getattr(t, "type", "")),
            ",".join(robots),
            ",".join(frames),
        ]) + "\n").encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _build_problem(plan_json, mp_tolerance):
    """Rebuild the exported mr_bench problem from a saved plan JSON alone.

    Mirrors main.py's --motion-plan branch, minus the two fuse passes (the raw paths were
    generated with --no-fuse-cograsp --no-fuse-pre-skill precisely so that this rebuild is
    exact: those passes rewrite the witnesses / inject pre-witnesses that the plan file does
    not carry)."""
    import assembly as A
    import replay as R
    import mp_export as MP
    from plan_io import read_plan, read_witnesses
    from solver import RobotIK

    cfg, plan = read_plan(plan_json)
    plan = [tuple(a) for a in plan]
    C, asm = R.build_scene(cfg)
    robots = A.get_robots(C)
    gk = A.wrench_kind_arg(C, robots) if cfg.get("wrench") else None

    rik = RobotIK(C, robots)
    rik.ik_seeding = True if cfg.get("ik_seeding") is None else bool(cfg.get("ik_seeding"))
    rik.pre_insertion_delta = float(cfg.get("pre_insertion") or 0.0)
    rik.pre_pick_delta = float(cfg.get("pre_pick") or 0.0)
    rik.pre_place_delta = float(cfg.get("pre_place") or 0.0)
    rik.pre_add_grasp_delta = float(cfg.get("pre_add_grasp") or 0.0)
    if cfg.get("mate_eq_weight") is not None:
        rik.mate_eq_weight = float(cfg["mate_eq_weight"])
    if cfg.get("place_eq_weight") is not None:
        rik.place_eq_weight = float(cfg["place_eq_weight"])
    if cfg.get("mate_accuracy") is not None:
        rik.mate_residual_reject = float(cfg["mate_accuracy"])
    if cfg.get("ik_accuracy") is not None:
        rik.nlp_stop_tolerance = float(cfg["ik_accuracy"])
    if cfg.get("witness_margin") is not None:
        rik.clearance_reject = float(cfg["witness_margin"])
    if cfg.get("witness_robot_collisions"):
        rik.strip_inactive_collision = False
    if cfg.get("strict_collisions") is not None:
        rik.strip_inactive_collision = False
        rik.penetration_reject = float(cfg["strict_collisions"])
        rik.col_weight = 1e2

    wit = read_witnesses(plan_json, list(rik.C.getJointNames()), len(plan))
    if wit is None:
        raise RuntimeError(
            f"{plan_json} carries no witness block. The benchmark pairs are generated with "
            f"--save AFTER the witness pass exactly so no re-solve is needed here; "
            f"regenerate the pair with scratchpad/gen_raw.sh.")

    prob = MP.make_problem(
        C, asm, plan, wit, robots, rik, gk,
        bracing=bool(cfg.get("bracing")),
        depgraph=not (True if cfg.get("mp_sequence") is None else bool(cfg["mp_sequence"])),
        collision_tolerance=float(mp_tolerance),
        pre_witnesses=None,
        robot_robot_margin=float(cfg.get("mp_robot_margin") or 0.0),
        robot_margin_ramp=float(cfg.get("mp_margin_ramp") or 0.0),
        table_margin=float(cfg.get("mp_table_margin") or 0.0),
        table_float=(float(cfg.get("part_float") or 0.0) if cfg.get("mp_table_margin") else 0.0))
    return prob, cfg, plan, MP


def _load_real_path(prob, MP, path_json):
    """Saved path -> real mr_bench States with the REBUILT problem's own Mode objects."""
    from multi_robot_multi_goal_planning.problems.planning_env import State
    saved = MP.load_path(path_json)
    want = getattr(MP.load_path, "last_signature", None)
    have = MP.task_signature(prob)

    chain = MP.mode_sequence(prob)
    by_ids = {}
    for m in chain:
        by_ids.setdefault(tuple(int(t) for t in np.atleast_1d(m.task_ids)), m)

    conf = type(prob.get_start_pos())
    sl = prob.get_start_pos()._array_slice
    out = []
    for s in saved:
        key = tuple(int(t) for t in np.atleast_1d(s.mode.task_ids))
        if key not in by_ids:
            raise RuntimeError(f"mode {key} of {path_json} is not in the rebuilt mode chain -- "
                               f"plan/path pair is mismatched")
        out.append(State(conf(np.asarray(s.q, dtype=float), sl), by_ids[key]))
    return out, want, have


def load_scene(name, verbose=True):
    """-> (env, raw_path, meta).  ONE per process (see module docstring)."""
    if _BUILT:
        raise RuntimeError(f"load_scene({name!r}): {_BUILT[0]!r} was already built in this "
                           f"process; assembly leaks rai scene globals between builds. "
                           f"Run one scene per process.")
    _BUILT.append(name)
    side = scene_meta(name)
    prob, cfg, plan, MP = _build_problem(side["plan"], side["mp_collision_tolerance"])
    raw, want_sig, _ = _load_real_path(prob, MP, side["raw_path"])
    have_sig = stable_task_signature(prob)

    meta = dict(side)
    meta.update(
        assembly=cfg.get("assembly"), parts=cfg.get("parts"),
        robot=cfg.get("robot"), grippers=cfg.get("grippers"),
        robots=list(prob.robots),
        robot_dims=[int(prob.robot_dims[r]) for r in prob.robots],
        n_tasks=len(prob.tasks), n_actions=len(plan),
        # what the path was PLANNED with -- read back, never assumed
        collision_resolution=float(prob.collision_resolution),
        collision_tolerance=float(prob.collision_tolerance),
        mp_robot_margin=float(cfg.get("mp_robot_margin") or 0.0),
        mp_margin_ramp=float(cfg.get("mp_margin_ramp") or 0.0),
        mp_table_margin=float(cfg.get("mp_table_margin") or 0.0),
        part_float=float(cfg.get("part_float") or 0.0),
        strict_collisions=cfg.get("strict_collisions"),
        task_signature_saved=want_sig, task_signature_rebuilt=have_sig,
        task_signature_ok=(want_sig is None or want_sig == have_sig),
    )
    meta.update(path_metrics(prob, raw))
    if not meta["task_signature_ok"]:
        raise RuntimeError(f"task signature mismatch for {name}: saved {want_sig} != "
                           f"rebuilt {have_sig}")
    if verbose:
        print(f"[benchenv] {name}: {meta['n_states']} states, {len(prob.robots)} robots, "
              f"{meta['n_modes']} modes, cost {meta['cost']:.4f}, "
              f"resolution {meta['collision_resolution']}, tolerance {meta['collision_tolerance']}")
    return prob, raw, meta


# ---------------------------------------------------------------------------
# the eligible interval space (OUR multi-robot extension)
# ---------------------------------------------------------------------------
def task_runs(path, r):
    """Maximal runs of constant task_ids[r] as [(i, j, task_id), ...] with i < j inclusive.

    robot_mode_shortcut only accepts an (i, j) pair when path[i].mode.task_ids[r] ==
    path[j].mode.task_ids[r], so THIS is robot r's eligible interval space: the union of the
    triangles {(a,b) : i <= a < b <= j} over these runs -- not the whole path triangle."""
    tids = [int(s.mode.task_ids[r]) for s in path]
    out, a = [], 0
    for b in range(len(tids)):
        if b + 1 < len(tids) and tids[b + 1] == tids[a]:
            continue
        if b > a:
            out.append((a, b, tids[a]))
        a = b + 1
    return out


def is_candidate(path, r, i, j, tol=1e-9):
    """Paper Sec. 4 / Definition 2, lifted to one robot: (i, j) is a CANDIDATE shortcut only
    if robot r's sub-path is not already a geodesic. Straight sub-paths can never improve
    cost, so discarding them before any collision check is free speed.

    Cheap and exact for the straight-line metric: arclength > chord, up to `tol`."""
    q = np.array([path[k].q[r] for k in range(i, j + 1)])
    if len(q) < 3:
        return False
    arc = float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
    return arc - float(np.linalg.norm(q[-1] - q[0])) > tol


def eligible_fraction(env, path):
    """Fraction of uniformly drawn (i, j, robot) triples that survive the task-id gate --
    i.e. the share of robot_mode_shortcut's attempts that are not thrown away for free."""
    n = len(path)
    tot = n * (n - 1) / 2.0
    per = []
    for r in range(len(env.robots)):
        ok = sum((j - i + 1) * (j - i) / 2.0 for i, j, _ in task_runs(path, r))
        per.append(ok / tot if tot else 0.0)
    return per


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def robot_length(path, r):
    q = np.array([path[k].q[r] for k in range(len(path))])
    return float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))) if len(q) > 1 else 0.0


def robot_detour(path, r):
    """Arclength minus straight-line, summed over robot r's maximal constant-task runs.
    This is the slack a per-robot shortcutter can actually remove: within a run the two
    endpoints are reachable by straight interpolation as far as the eligibility gate is
    concerned, so (length - |q_j - q_i|) upper-bounds the achievable saving on that run."""
    tot = 0.0
    for i, j, _ in task_runs(path, r):
        q = np.array([path[k].q[r] for k in range(i, j + 1)])
        tot += (float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
                - float(np.linalg.norm(q[-1] - q[0])))
    return tot


def path_metrics(env, path):
    from multi_robot_multi_goal_planning.problems.util import path_cost
    R = len(env.robots)
    modes = [tuple(int(t) for t in np.atleast_1d(s.mode.task_ids)) for s in path]
    nmodes = 1 + sum(1 for k in range(1, len(modes)) if modes[k] != modes[k - 1])
    lens = [robot_length(path, r) for r in range(R)]
    det = [robot_detour(path, r) for r in range(R)]
    return dict(
        n_states=len(path), n_modes=nmodes,
        cost=float(path_cost(path, env.batch_config_cost)),
        per_robot_length=lens, per_robot_detour=det,
        total_length=float(sum(lens)), total_detour=float(sum(det)),
        n_runs=[len(task_runs(path, r)) for r in range(R)],
        run_lengths=[[j - i + 1 for i, j, _ in task_runs(path, r)] for r in range(R)],
        eligible_fraction=eligible_fraction(env, path),
    )


# ---------------------------------------------------------------------------
# the benchmark's x-axis: number of collision checks
# ---------------------------------------------------------------------------
class CheckCounter:
    """Counts collision checks on ONE env instance (mr_bench stays untouched: this rebinds
    bound methods on the instance, on top of whatever mp_export's margin wrapper installed).

    `.configs` is the paper's axis -- one per validated configuration.  `.edges` / `.paths`
    are the coarser call counts, kept because a per-edge accounting is sometimes wanted."""

    def __init__(self, env):
        self.env, self.configs, self.edges, self.paths = env, 0, 0, 0
        self._orig = {}
        for nm in ("is_collision_free", "is_collision_free_np",
                   "is_edge_collision_free", "is_path_collision_free"):
            self._orig[nm] = getattr(env, nm)
        env.is_collision_free = self._wrap_cfg(self._orig["is_collision_free"])
        env.is_collision_free_np = self._wrap_cfg(self._orig["is_collision_free_np"])
        env.is_edge_collision_free = self._wrap("edges", self._orig["is_edge_collision_free"])
        env.is_path_collision_free = self._wrap("paths", self._orig["is_path_collision_free"])

    def _wrap_cfg(self, fn):
        def g(*a, **kw):
            self.configs += 1
            return fn(*a, **kw)
        return g

    def _wrap(self, field, fn):
        def g(*a, **kw):
            setattr(self, field, getattr(self, field) + 1)
            return fn(*a, **kw)
        return g

    def reset(self):
        self.configs = self.edges = self.paths = 0
        return self

    def restore(self):
        for nm, fn in self._orig.items():
            setattr(self.env, nm, fn)

    def __repr__(self):
        return f"<checks configs={self.configs} edges={self.edges} paths={self.paths}>"


def count_collision_checks(env):
    return CheckCounter(env)


# ---------------------------------------------------------------------------
# the exact input robot_mode_shortcut would start from
# ---------------------------------------------------------------------------
def prepared_path(env, path, interpolation_resolution=0.5):
    """robot_mode_shortcut's first two lines: drop collinear nodes, then re-interpolate.
    Every shortcutting variant must start from THIS so cost curves are comparable."""
    from multi_robot_multi_goal_planning.problems.util import interpolate_path
    from multi_robot_multi_goal_planning.planners.shortcutting import remove_interpolated_nodes
    return interpolate_path(remove_interpolated_nodes(path), interpolation_resolution)


def is_path_valid(env, path):
    """Every state of `path` collision-free under the env the path was planned in."""
    return [k for k in range(len(path)) if not env.is_collision_free(path[k].q, path[k].mode)]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint
    if len(sys.argv) < 2:
        print("scenes:", ", ".join(scene_names()))
        raise SystemExit(0)
    nm = sys.argv[1]
    env, raw, meta = load_scene(nm)
    prep = prepared_path(env, raw)
    meta["prepared"] = path_metrics(env, prep)
    bad_raw = is_path_valid(env, raw)
    bad_prep = is_path_valid(env, prep)
    meta["raw_colliding_states"] = bad_raw
    meta["prepared_colliding_states"] = bad_prep
    pprint.pprint(meta, sort_dicts=True, width=110)
    with open(os.path.join(RAW, f"{nm}-inventory.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print("wrote", os.path.join(RAW, f"{nm}-inventory.json"))
